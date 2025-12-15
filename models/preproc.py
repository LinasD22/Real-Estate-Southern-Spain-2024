import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import Dict, List
import joblib
import torch
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
print("LOADED preproc.py")

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# TODO if dataset iamges and decriptions are not inside these folders add
_models_dir = Path(__file__).parent
_data_dir = _models_dir.parent / "data"
DESCRIPTIONS_DIR = _data_dir / "descriptions"
IMAGES_DIR = _data_dir / "images"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # 512-dim

def load_description(ref: str) -> str:
    file_path = DESCRIPTIONS_DIR / f"{ref}.txt"
    if file_path.exists():
        return file_path.read_text(encoding="utf-8", errors="ignore")
    return "" 

def encode_descriptions():
    models_dir = Path(__file__).parent
    data_dir = models_dir.parent / "data"

    csv_path = (data_dir / "properties.csv").resolve()
    df = pd.read_csv(csv_path)
    df["reference"] = df["reference"].astype(str)

    df["description"] = df["reference"].apply(load_description)

    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    texts = df["description"].tolist()

    X_text = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    emb_path = data_dir / "description_embeddings.npy"
    np.save(emb_path, X_text)

    return df, X_text

def load_images_for_reference(ref: str):
    ref_dir = IMAGES_DIR / ref
    if not ref_dir.exists() or not ref_dir.is_dir():
        return []

    images = []
    for img_path in ref_dir.iterdir():
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception:
                continue
    return images
	

def encode_images():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_dir = Path(__file__).parent
    data_dir = models_dir.parent / "data"

    df = pd.read_csv((data_dir / "properties.csv").resolve())
    df["reference"] = df["reference"].astype(str)

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    all_embeddings = []

    for ref in tqdm(df["reference"], desc="Encoding images"):
        images = load_images_for_reference(ref)

        if not images:
            all_embeddings.append(np.zeros(512))
            continue

        inputs = clip_processor(
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.cpu().numpy()

        all_embeddings.append(emb.mean(axis=0))

    X_img = np.vstack(all_embeddings)
    np.save(data_dir / "image_embeddings.npy", X_img)

    return df, X_img


     
def expand_img_embeddings(df_, prefix="img_emb"):
    emb = np.vstack(df_["_img_emb"].values)
    cols = [f"{prefix}_{i}" for i in range(emb.shape[1])]
    df_[cols] = emb
    return cols

def expand_embeddings(df_, prefix="emb"):
    emb = np.vstack(df_["_text_emb"].values)
    cols = [f"{prefix}_{i}" for i in range(emb.shape[1])]
    df_[cols] = emb
    return cols


def _apply_caps(df, limit):
	df = df.copy()
	for col, cap in limit.items():
		if col in df.columns:
			df[col] = df[col].clip(upper=cap)
	return df

def get_preprocessed_data():
    models_dir = Path(__file__).parent
    data_dir = models_dir.parent / "data"

    csv_path = (data_dir / "properties.csv").resolve()
    emb_path = (data_dir / "description_embeddings.npy").resolve()

    df = pd.read_csv(csv_path)
    df = df.copy()
    df.columns = df.columns.str.strip()

    X_text = np.load(emb_path)
    assert len(df) == X_text.shape[0], "Mismatch CSV vs embeddings"
    df["_text_emb"] = list(X_text)

    img_emb_path = (data_dir / "image_embeddings.npy").resolve()
    X_img = np.load(img_emb_path)

    assert len(df) == X_img.shape[0], "Mismatch CSV vs image embeddings"
    df["_img_emb"] = list(X_img)

    # Price cleanup and target
    df["price"] = (
        df["price"].astype(str)
        .str.replace(r"[^\d.,]", "", regex=True)
        .str.replace(",", "")
        .astype(float))

    # Features list
    feat_col = "pipe-separed list of features of the property"
    if feat_col in df.columns:
        df["features"] = df[feat_col].astype(str).str.lower().str.split("|")
    else:
        df["features"] = [[] for _ in range(len(df))]

    # Property type derived from title
    if "title" in df.columns:
        df["property_type"] = (
            df["title"].astype(str)
            .str.replace(r"^\d+\s*Bedroom(s)?\s*", "", regex=True)
            .str.replace(r"^\d+\s*", "", regex=True)
            .str.strip()
        )
    else:
        df["property_type"] = "Unknown"

    # Bedrooms/Bathrooms missing for non-residential
    non_residential = {
        "Plot", "Land", "Commercial Property", "Commercial Development",
        "Warehouse", "Shop", "Office", "Restaurant", "Bar / Nightclub",
        "Cafe", "Parking", "Garage", "Storage", "Property", "Other",
    }
    if "property_type" in df.columns:
        mask_non_res = df["property_type"].isin(non_residential)
        for col in ["bedrooms", "bathrooms"]:
            if col in df.columns:
                df.loc[mask_non_res, col] = np.nan

    # Split location into city/region
    if "location" in df.columns:
        df["location"] = df["location"].astype(str).str.strip()
        split = df["location"].str.split(",", n=1, expand=True)
        df["city"] = split[0].str.strip()
        df["region"] = split[1].str.strip() if split.shape[1] > 1 else ""
    else:
        df["city"] = ""
        df["region"] = ""

    # Train/Val/Test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    tr, val = train_test_split(train_df, test_size=0.2, random_state=42)

    # Caps for extreme values
    limit: Dict[str, float] = {}
    for col in ["price", "indoor surface area in sqm", "outdoor surface area in sqm"]:
        if col in tr.columns:
            limit[col] = tr[col].quantile(0.995)
    for col in ["bedrooms", "bathrooms"]:
        if col in tr.columns:
            limit[col] = tr[col].quantile(0.99)

    tr = _apply_caps(tr, limit)
    val = _apply_caps(val, limit)
    test_df = _apply_caps(test_df, limit)

    # Target mean for city
    tr["y"] = np.log(tr["price"])
    global_mean = tr["y"].mean()
    city_stats = tr.groupby("city")["y"].agg(["mean", "count"])
    k = 20
    city_stats["mean"] = (city_stats["count"] * city_stats["mean"] + k * global_mean) / (
        city_stats["count"] + k
    )

    def add_city_mean(df_):
        return df_.assign(city_mean=df_["city"].map(city_stats["mean"]).fillna(global_mean))

    tr = add_city_mean(tr)
    val = add_city_mean(val)
    test_df = add_city_mean(test_df)

    # One-hot encode property_type
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(tr[["property_type"]])
    ohe_cols = list(ohe.get_feature_names_out(["property_type"]))

    def apply_ohe(df_):
        arr = ohe.transform(df_[["property_type"]])
        df_[ohe_cols] = arr
        return df_

    tr = apply_ohe(tr)
    val = apply_ohe(val)
    test_df = apply_ohe(test_df)

    # Top features flags
    counts = Counter(pd.Series(sum(tr["features"].tolist(), [])))
    feature_names = [name for name, _ in counts.most_common(30)]

    for f in feature_names:
        for d in (tr, val, test_df):
            d[f] = d["features"].apply(lambda x: int(f in x) if isinstance(x, list) else 0)


    emb_cols = expand_embeddings(tr)
    expand_embeddings(val)
    expand_embeddings(test_df)

    img_emb_cols = expand_img_embeddings(tr)
    expand_img_embeddings(val)
    expand_img_embeddings(test_df)

    # Final features
    base_cols = [c for c in ["city_mean", "bedrooms", "bathrooms"] if c in tr.columns]
    feature_cols = base_cols + feature_names + ohe_cols + emb_cols + img_emb_cols

    X_tr = tr[feature_cols]
    X_val = val[feature_cols]
    X_test = test_df[feature_cols]

    y_tr = np.log1p(tr["price"])
    y_val = np.log1p(val["price"])

    joblib.dump(
    {
        "ohe": ohe,
        "limit": limit,
        "city_stats": city_stats,
        "global_mean": global_mean,
        "k": k,

        # Feature structure
        "feature_cols": feature_cols,
        "base_cols": base_cols,
        "ohe_cols": ohe_cols,
        "top_features": feature_names,

        # Text embeddings
        "text_embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "text_embedding_dim": X_text.shape[1],

        # Image embeddings
        "clip_model_name": CLIP_MODEL_NAME,
        "image_embedding_dim": len(img_emb_cols),
    },
    models_dir / "preprocessing.joblib",
)
    out_dir = models_dir.parent / "data" 
    out_dir.mkdir(exist_ok=True)
    X_tr.to_csv(out_dir / "X_tr.csv", index=False)
    X_val.to_csv(out_dir / "X_val.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_tr.to_csv(out_dir / "y_tr.csv", index=False)
    y_val.to_csv(out_dir / "y_val.csv", index=False)

    return X_tr, y_tr, X_val, y_val, X_test, feature_cols


