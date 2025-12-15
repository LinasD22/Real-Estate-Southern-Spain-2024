import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
from PIL import Image
from sentence_transformers import SentenceTransformer
import io

torch = None
CLIPModel = None
CLIPProcessor = None

def _ensure_torch():
    global torch, CLIPModel, CLIPProcessor
    if torch is None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            st.error("torch and transformers are required for image encoding. Please install: pip install torch transformers")
            return False
    return True

device = None

def _get_device():
    global device
    if device is None:
        if _ensure_torch():
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"
    return device

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


@st.cache_resource
def load_xgboost_model():
    model_path = Path(__file__).parent.parent / "xgboost_model.json"
    if not model_path.exists():
        return None

    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(str(model_path))
    return model


@st.cache_resource
def load_preprocessing_pipeline():
    pipeline_path = Path(__file__).parent.parent / "models" / "preprocessing.joblib"
    if not pipeline_path.exists():
        return None

    return joblib.load(pipeline_path)


@st.cache_resource
def load_clip_model():
    if not _ensure_torch():
        return None, None

    try:
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_get_device()).eval()
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        return model, processor
    except Exception as e:
        st.error(f"Failed to load CLIP model: {e}")
        return None, None


@st.cache_resource
def load_text_encoder():
    try:
        return SentenceTransformer(TEXT_EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Failed to load text encoder: {e}")
        return None


def encode_images_to_embeddings(image_data_list: List[Union[bytes, Any]]) -> np.ndarray:

    if not image_data_list:
        # Return zero vector if no images
        return np.zeros(512)

    if not _ensure_torch():
        return np.zeros(512)

    clip_model, clip_processor = load_clip_model()
    if clip_model is None or clip_processor is None:
        st.warning("CLIP model not loaded. Using zero embeddings for images.")
        return np.zeros(512)

    try:
        images = []
        for idx, img_data in enumerate(image_data_list):
            try:
                if isinstance(img_data, bytes):
                    if len(img_data) == 0:
                        st.warning(f"Image {idx + 1} is empty, skipping...")
                        continue
                    img_bytes_io = io.BytesIO(img_data)
                    img = Image.open(img_bytes_io).convert("RGB")
                else:
                    img = Image.open(img_data).convert("RGB")
                images.append(img)
            except Exception as img_error:
                st.warning(f"Error processing image {idx + 1}: {img_error}. Skipping this image.")
                continue
        
        if not images:
            st.warning("No valid images could be processed. Using zero embeddings.")
            return np.zeros(512)

        # Process all images
        inputs = clip_processor(
            images=images,
            return_tensors="pt",
            padding=True
        ).to(_get_device())

        # Get embeddings
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.cpu().numpy()

        # Average embeddings across images
        return emb.mean(axis=0)

    except Exception as e:
        st.error(f"Error encoding images: {e}")
        return np.zeros(512)


def encode_description_to_embedding(description: str) -> np.ndarray:

    if not description or not description.strip():
        text_encoder = load_text_encoder()
        if text_encoder is None:
            return np.zeros(768)  # Default dimension for all-mpnet-base-v2
        return text_encoder.encode("")

    text_encoder = load_text_encoder()
    if text_encoder is None:
        st.warning("Text encoder not loaded. Using zero embeddings for description.")
        return np.zeros(768)

    try:
        return text_encoder.encode(description)
    except Exception as e:
        st.error(f"Error encoding description: {e}")
        return np.zeros(768)


def prepare_features_for_prediction(
    property_data: Dict,
    pipeline: Dict,
    image_embeddings: Optional[np.ndarray] = None,
    description_embeddings: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool]:

    if pipeline is None:
        st.error("Preprocessing pipeline not loaded.")
        return None, False

    try:
        # Extract required metadata from pipeline
        ohe = pipeline.get("ohe")
        feature_cols = pipeline.get("feature_cols", [])
        base_cols = pipeline.get("base_cols", [])
        ohe_cols = pipeline.get("ohe_cols", [])
        top_features = pipeline.get("top_features", [])

        feature_vector = {}

        city = property_data.get("location", "").split(",")[0].strip() if property_data.get("location") else ""
        city_stats = pipeline.get("city_stats")
        global_mean = pipeline.get("global_mean", 0)
        k = pipeline.get("k", 20)

        if city_stats is not None and city in city_stats.index:
            city_mean = city_stats.loc[city, "mean"]
        else:
            city_mean = global_mean

        feature_vector["city_mean"] = city_mean
        feature_vector["bedrooms"] = property_data.get("bedrooms", 0)
        feature_vector["bathrooms"] = property_data.get("bathrooms", 0)

        features_list = property_data.get("features_list", [])
        if isinstance(features_list, str):
            features_list = [f.strip() for f in features_list.split("|")]

        for feat in top_features:
            feature_vector[feat] = 1 if feat in features_list else 0

        property_type = property_data.get("property_type", "Unknown")
        if ohe is not None:
            try:
                ohe_input = pd.DataFrame([[property_type]], columns=["property_type"])
                ohe_arr = ohe.transform(ohe_input)[0]
                for col, val in zip(ohe_cols, ohe_arr):
                    feature_vector[col] = val
            except Exception:
                for col in ohe_cols:
                    feature_vector[col] = 0

        if description_embeddings is not None:
            for i, val in enumerate(description_embeddings):
                feature_vector[f"emb_{i}"] = val
        else:
            desc_emb_dim = pipeline.get("text_embedding_dim", 768)
            for i in range(desc_emb_dim):
                feature_vector[f"emb_{i}"] = 0

        # 5. Image embeddings
        if image_embeddings is not None:
            for i, val in enumerate(image_embeddings):
                feature_vector[f"img_emb_{i}"] = val
        else:
            img_emb_dim = pipeline.get("image_embedding_dim", 512)
            for i in range(img_emb_dim):
                feature_vector[f"img_emb_{i}"] = 0

        feature_array = np.array([
            [feature_vector.get(col, 0) for col in feature_cols]
        ], dtype=np.float32)

        return feature_array, True

    except Exception as e:
        st.error(f"Error preparing features: {e}")
        return None, False


def make_prediction(
    property_data: Dict,
    image_data_list: Optional[List[bytes]] = None,
    description_text: Optional[str] = None
) -> Optional[float]:

    pipeline = load_preprocessing_pipeline()
    model = load_xgboost_model()

    if pipeline is None:
        st.error("Preprocessing pipeline not found.")
        return None

    if model is None:
        st.error("XGBoost model not found.")
        return None

    if image_data_list:
        image_emb = encode_images_to_embeddings(image_data_list)
    else:
        image_emb = None

    if description_text:
        desc_emb = encode_description_to_embedding(description_text)
    else:
        desc_emb = None

    X_tabular, success = prepare_features_for_prediction(
        property_data,
        pipeline,
        image_embeddings=image_emb,
        description_embeddings=desc_emb
    )

    if not success or X_tabular is None:
        st.error("Failed to prepare features for prediction.")
        return None

    parts = [X_tabular]
    if desc_emb is not None:
        parts.append(desc_emb.reshape(1, -1))
    if image_emb is not None:
        parts.append(image_emb.reshape(1, -1))
    
    X = np.concatenate(parts, axis=1)

    try:
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        y_pred_log = model.predict(dmatrix)[0]
        y_pred = np.expm1(y_pred_log)
        return max(0, y_pred)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


def get_prediction_with_metadata(
    property_data: Dict,
    image_data_list: Optional[List[bytes]] = None,
    description_text: Optional[str] = None
) -> Optional[Dict]:

    prediction = make_prediction(property_data, image_data_list, description_text)

    if prediction is None:
        return None

    margin = prediction * 0.15

    return {
        "predicted_price": prediction,
        "confidence_lower": prediction - margin,
        "confidence_upper": prediction + margin,
        "images_provided": len(image_data_list) > 0 if image_data_list else False,
        "description_provided": bool(description_text),
    }

