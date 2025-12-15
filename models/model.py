from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import json
from datetime import datetime


def _detect_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "price",
        "price_eur",
        "target",
        "y",
        "sale_price",
        "sold_price",
        "label",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_preprocessed_data(data_dir: str = "data", models_dir: Optional[str] = None) -> Dict[str, object]:
    data_dir = Path(data_dir)
    if models_dir is None:
        models_dir = Path(__file__).parent
    else:
        models_dir = Path(models_dir)

    filenames = {
        "X_tr": "X_tr.csv",
        "X_val": "X_val.csv",
        "X_test": "X_test.csv",
        "y_tr": "y_tr.csv",
        "y_val": "y_val.csv",
        "preproc_joblib": "preprocessing.joblib",
        "description_embeddings": "description_embeddings.npy",
        "image_embeddings": "image_embeddings.npy",
    }

    candidate_dirs = [data_dir]
    alt = data_dir / "Data"
    if alt.exists() and alt.is_dir():
        candidate_dirs.append(alt)

    result: Dict[str, object] = {}

    for key, fname in filenames.items():
        value = None
        for cand in candidate_dirs:
            path = cand / fname
            if path.exists():
                if path.suffix == ".csv":
                    df = pd.read_csv(path)
                    if key.startswith("y_"):
                        value = df.iloc[:, 0].values
                    else:
                        value = df
                elif path.suffix == ".npy":
                    value = np.load(path)
                elif path.suffix == ".joblib":
                    value = joblib.load(path)
                break
        result[key] = value

    return result


def combine_all_data(data: Dict[str, object]) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    X_parts = []
    y_parts = []

    if data.get('X_tr') is not None:
        X_parts.append(data['X_tr'])
        if data.get('y_tr') is not None:
            y_parts.append(data['y_tr'])
    if data.get('X_val') is not None:
        X_parts.append(data['X_val'])
        if data.get('y_val') is not None:
            y_parts.append(data['y_val'])
    if data.get('X_test') is not None:
        X_parts.append(data['X_test'])
    if not X_parts:
        return pd.DataFrame(), np.array([]), None, None

    X_all = pd.concat(X_parts, axis=0, ignore_index=True)
    y_all = np.concatenate(y_parts) if y_parts else None

    desc_embeddings_all = None
    img_embeddings_all = None
    if data.get('description_embeddings') is not None:
        desc = data['description_embeddings']
        desc_embeddings_all = desc[: len(X_all)] if len(desc) >= len(X_all) else desc
    if data.get('image_embeddings') is not None:
        img = data['image_embeddings']
        img_embeddings_all = img[: len(X_all)] if len(img) >= len(X_all) else img

    return X_all, y_all, desc_embeddings_all, img_embeddings_all


def split_data(X: pd.DataFrame, y: np.ndarray,
               desc_embeddings: Optional[np.ndarray],
               img_embeddings: Optional[np.ndarray],
               train_pct: float = 90.0,
               test_pct: float = 10.0,
               val_pct: float = 10.0,
               random_state: int = 42) -> Dict[str, object]:
    assert abs(train_pct + test_pct - 100.0) < 1e-6, "train_pct + test_pct must equal 100"
    assert 0 <= val_pct < train_pct, "val_pct must be between 0 and train_pct"

    n_samples = len(X)
    result: Dict[str, object] = {}

    if test_pct == 0:
        X_train_full = X.reset_index(drop=True)
        y_train_full = y
        desc_emb_train_full = desc_embeddings
        img_emb_train_full = img_embeddings
        result['X_test'] = None
        result['y_test'] = None
        result['desc_embeddings_test'] = None
        result['img_embeddings_test'] = None
    else:
        test_size = test_pct / 100.0
        indices = np.arange(n_samples)
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

        X_train_full = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train_full = y[train_idx]
        y_test = y[test_idx]

        result['X_test'] = X_test
        result['y_test'] = y_test

        if desc_embeddings is not None:
            desc_emb_train_full = desc_embeddings[train_idx]
            result['desc_embeddings_test'] = desc_embeddings[test_idx]
        else:
            desc_emb_train_full = None
            result['desc_embeddings_test'] = None

        if img_embeddings is not None:
            img_emb_train_full = img_embeddings[train_idx]
            result['img_embeddings_test'] = img_embeddings[test_idx]
        else:
            img_emb_train_full = None
            result['img_embeddings_test'] = None

    if val_pct > 0:
        val_size = val_pct / train_pct
        train_indices = np.arange(len(X_train_full))
        train_idx, val_idx = train_test_split(train_indices, test_size=val_size, random_state=random_state)

        result['X_train'] = X_train_full.iloc[train_idx].reset_index(drop=True)
        result['X_val'] = X_train_full.iloc[val_idx].reset_index(drop=True)
        result['y_train'] = y_train_full[train_idx]
        result['y_val'] = y_train_full[val_idx]

        if desc_emb_train_full is not None:
            result['desc_embeddings_train'] = desc_emb_train_full[train_idx]
            result['desc_embeddings_val'] = desc_emb_train_full[val_idx]
        else:
            result['desc_embeddings_train'] = None
            result['desc_embeddings_val'] = None

        if img_emb_train_full is not None:
            result['img_embeddings_train'] = img_emb_train_full[train_idx]
            result['img_embeddings_val'] = img_emb_train_full[val_idx]
        else:
            result['img_embeddings_train'] = None
            result['img_embeddings_val'] = None
    else:
        result['X_train'] = X_train_full
        result['X_val'] = None
        result['y_train'] = y_train_full
        result['y_val'] = None
        result['desc_embeddings_train'] = desc_emb_train_full
        result['desc_embeddings_val'] = None
        result['img_embeddings_train'] = img_emb_train_full
        result['img_embeddings_val'] = None

    return result


def combine_features(X_tabular: pd.DataFrame,
                     desc_embeddings: Optional[np.ndarray] = None,
                     img_embeddings: Optional[np.ndarray] = None) -> np.ndarray:
    parts = [X_tabular.values]
    if desc_embeddings is not None:
        parts.append(desc_embeddings)
    if img_embeddings is not None:
        parts.append(img_embeddings)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=1)


def train_xgboost_model(splits: Dict[str, object],
                       params: Optional[Dict] = None,
                       save_model: bool = True,
                       model_path: str = "xgboost_model.json") -> xgb.XGBRegressor:
    if params is None:
        custom_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,
            'gamma': 0.2,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'n_estimators': 800,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
        }
    else:
        custom_params = params.copy()
    
    X_train = combine_features(
        splits['X_train'],
        splits.get('desc_embeddings_train'),
        splits.get('img_embeddings_train')
    )
    y_train = splits['y_train']

    has_validation = splits.get('X_val') is not None and splits.get('y_val') is not None
    if has_validation:
        X_val = combine_features(
            splits['X_val'],
            splits.get('desc_embeddings_val'),
            splits.get('img_embeddings_val')
        )
        y_val = splits['y_val']

    early_stopping_rounds = custom_params.pop('early_stopping_rounds', None)
    custom_params.pop('eval_metric', None)

    if has_validation and early_stopping_rounds:
        custom_params['early_stopping_rounds'] = early_stopping_rounds

    model = xgb.XGBRegressor(**custom_params)

    if has_validation:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    else:
        model.fit(X_train, y_train, verbose=100)

    y_train_pred = model.predict(X_train)
    print("\n" + "=" * 50)
    print("TRAINING METRICS:")
    print(f"RMSE:  {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
    print(f"MAE:   {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"R²:    {r2_score(y_train, y_train_pred):.4f}")

    if has_validation:
        y_val_pred = model.predict(X_val)
        print("VALIDATION METRICS:")
        print(f"RMSE:  {np.sqrt(mean_squared_error(y_val, y_val_pred)):.4f}")
        print(f"MAE:   {mean_absolute_error(y_val, y_val_pred):.4f}")
        print(f"R²:    {r2_score(y_val, y_val_pred):.4f}")

    if save_model:
        model.save_model(model_path)
        print(f"Model saved to {model_path}")

    return model


def evaluate_on_test(model: xgb.XGBRegressor, splits: Dict[str, object]) -> Optional[pd.DataFrame]:
    if splits.get('X_test') is None or splits.get('y_test') is None:
        print("\nNo test set available - skipping test evaluation")
        return None

    X_test = combine_features(
        splits['X_test'],
        splits.get('desc_embeddings_test'),
        splits.get('img_embeddings_test')
    )
    y_test = splits['y_test']
    preds = model.predict(X_test)

    try:
        actual_prices = np.expm1(y_test)
        pred_prices = np.expm1(preds)
    except Exception:
        actual_prices = y_test
        pred_prices = preds

    df_out = pd.DataFrame({
        'actual_price': actual_prices,
        'predicted_price': pred_prices
    })

    print("=" * 50)
    print("TEST METRICS:")
    print(f"RMSE:  {np.sqrt(mean_squared_error(y_test, preds)):.4f}")
    print(f"MAE:   {mean_absolute_error(y_test, preds):.4f}")
    print(f"R²:    {r2_score(y_test, preds):.4f}")
    print("=" * 50)

    out_path = Path("test_predictions.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Test predictions saved to {out_path}")
    
    error = pred_prices - actual_prices
    error_pct = (error / actual_prices) * 100
    print(f"\nTest Error Analysis:")
    print(f"  Mean Absolute Error: €{abs(error).mean():.2f}")
    print(f"  Mean Absolute % Error: {abs(error_pct).mean():.2f}%")
    
    return df_out


def save_feature_metadata(splits: Dict[str, object], data: Dict[str, object]):
    """Save all metadata needed for frontend integration"""
    
    X_train = splits['X_train']
    
    metadata = {
        'model_version': '1.0',
        'trained_date': datetime.now().isoformat(),
        'target_variable': 'price_eur',
        'target_transform': 'log1p',
        'required_features': list(X_train.columns),
        'feature_info': {},
        'uses_description_embeddings': splits.get('desc_embeddings_train') is not None,
        'uses_image_embeddings': splits.get('img_embeddings_train') is not None,
        'description_embedding_dim': splits['desc_embeddings_train'].shape[1] if splits.get('desc_embeddings_train') is not None else 0,
        'image_embedding_dim': splits['img_embeddings_train'].shape[1] if splits.get('img_embeddings_train') is not None else 0,
    }
    
    # Extract feature information
    for col in X_train.columns:
        feature_info = {'type': str(X_train[col].dtype)}
        
        if X_train[col].dtype in ['int64', 'float64']:
            feature_info['min'] = float(X_train[col].min())
            feature_info['max'] = float(X_train[col].max())
            feature_info['mean'] = float(X_train[col].mean())
        elif X_train[col].dtype == 'object':
            unique_vals = X_train[col].unique().tolist()
            feature_info['valid_values'] = unique_vals[:50]  # Limit to 50 values
        
        metadata['feature_info'][col] = feature_info
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Model metadata saved to model_metadata.json")


def save_input_schema(splits: Dict[str, object]):
    """Save a clear input schema for frontend developers"""
    
    X_train = splits['X_train']
    
    # Identify one-hot encoded prefixes
    encoded_prefixes = set()
    for col in X_train.columns:
        if '_' in col and not col.startswith('desc_emb') and not col.startswith('img_emb'):
            prefix = col.rsplit('_', 1)[0]
            if sum(c.startswith(f"{prefix}_") for c in X_train.columns) > 1:
                encoded_prefixes.add(prefix)
    
    schema = {
        'description': 'Input schema for property price prediction',
        'raw_inputs': {},
        'notes': []
    }
    
    # Add raw input fields
    for col in X_train.columns:
        is_encoded = any(col.startswith(f"{prefix}_") for prefix in encoded_prefixes)
        if is_encoded or col.startswith('desc_emb') or col.startswith('img_emb'):
            continue
        
        field_schema = {
            'type': 'number' if X_train[col].dtype in ['int64', 'float64'] else 'string',
            'required': True
        }
        
        if X_train[col].dtype in ['int64', 'float64']:
            field_schema['example'] = float(X_train[col].median())
        
        schema['raw_inputs'][col] = field_schema
    
    # Add categorical options
    for prefix in encoded_prefixes:
        encoded_cols = [c for c in X_train.columns if c.startswith(f"{prefix}_")]
        categories = [c.replace(f"{prefix}_", "") for c in encoded_cols]
        
        schema['raw_inputs'][prefix] = {
            'type': 'string',
            'required': True,
            'valid_values': categories,
            'example': categories[0]
        }
    
    if splits.get('desc_embeddings_train') is not None:
        schema['notes'].append('Description text will be embedded on the backend')
    
    if splits.get('img_embeddings_train') is not None:
        schema['notes'].append('Images will be embedded on the backend')
    
    with open('api_input_schema.json', 'w') as f:
        json.dump(schema, f, indent=2)
    
    print("✓ API input schema saved to api_input_schema.json")


def save_model_report(model: xgb.XGBRegressor, splits: Dict[str, object]):
    """Save model performance metrics for documentation"""
    
    X_test = combine_features(
        splits['X_test'],
        splits.get('desc_embeddings_test'),
        splits.get('img_embeddings_test')
    )
    y_test = splits['y_test']
    y_pred = model.predict(X_test)
    
    report = {
        'model_type': 'XGBoost Regressor',
        'generated_at': datetime.now().isoformat(),
        'dataset_sizes': {
            'train': len(splits['X_train']),
            'validation': len(splits['X_val']),
            'test': len(splits['X_test'])
        },
        'performance': {
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'test_mae': float(mean_absolute_error(y_test, y_pred)),
            'test_r2': float(r2_score(y_test, y_pred))
        },
        'hyperparameters': model.get_params(),
        'notes': [
            'Prices are log-transformed. Use np.expm1() to convert predictions back.',
            'Model expects preprocessed features matching training schema.'
        ]
    }
    
    with open('model_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✓ Model report saved to model_report.json")


if __name__ == "__main__":
    TRAIN_PCT = 90
    TEST_PCT = 10
    VAL_PCT = 10
    RANDOM_STATE = 42

    custom_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.035,
        'max_depth': 5,
        'min_child_weight': 5,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'colsample_bylevel': 0.7,
        'gamma': 0.2,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'n_estimators': 800,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'early_stopping_rounds': 50,
    }

    print("Loading preprocessed data...")
    data = load_preprocessed_data()

    if data.get('X_tr') is not None and data.get('X_val') is not None:
        print("Using pre-saved splits and splitting validation set in half...")
        print("✅ TRAINING WITH ALL FEATURES (TABULAR + DESCRIPTION + IMAGE EMBEDDINGS)")
        
        n_tr = len(data['X_tr'])
        n_val = len(data['X_val'])
        
        val_indices = np.arange(n_val)
        val_idx, test_idx = train_test_split(val_indices, test_size=0.5, random_state=RANDOM_STATE)
        
        desc_emb = data.get('description_embeddings')
        img_emb = data.get('image_embeddings')
        X_val_full = data['X_val']
        y_val_full = data['y_val']
        
        splits = {
            'X_train': data['X_tr'].reset_index(drop=True),
            'X_val': X_val_full.iloc[val_idx].reset_index(drop=True),
            'X_test': X_val_full.iloc[test_idx].reset_index(drop=True),
            'y_train': data['y_tr'],
            'y_val': y_val_full[val_idx],
            'y_test': y_val_full[test_idx],
            'desc_embeddings_train': desc_emb[:n_tr] if desc_emb is not None and len(desc_emb) >= n_tr else None,
            'desc_embeddings_val': desc_emb[n_tr:n_tr+n_val][val_idx] if desc_emb is not None and len(desc_emb) >= n_tr+n_val else None,
            'desc_embeddings_test': desc_emb[n_tr:n_tr+n_val][test_idx] if desc_emb is not None and len(desc_emb) >= n_tr+n_val else None,
            'img_embeddings_train': img_emb[:n_tr] if img_emb is not None and len(img_emb) >= n_tr else None,
            'img_embeddings_val': img_emb[n_tr:n_tr+n_val][val_idx] if img_emb is not None and len(img_emb) >= n_tr+n_val else None,
            'img_embeddings_test': img_emb[n_tr:n_tr+n_val][test_idx] if img_emb is not None and len(img_emb) >= n_tr+n_val else None,
        }
    else:
        print("Combining all available data and splitting according to percentages...")
        print("✅ TRAINING WITH ALL FEATURES (TABULAR + DESCRIPTION + IMAGE EMBEDDINGS)")
        X_all, y_all, desc_emb_all, img_emb_all = combine_all_data(data)
        
        if X_all.empty or y_all is None or len(y_all) == 0:
            raise ValueError("No valid data found to train on!")
        
        print(f"Total samples: {len(X_all)}, Features: {X_all.shape[1]}")
        if desc_emb_all is not None:
            print(f"Description embedding dimensions: {desc_emb_all.shape[1]}")
        if img_emb_all is not None:
            print(f"Image embedding dimensions: {img_emb_all.shape[1]}")

        splits = split_data(
            X_all, y_all, desc_emb_all, img_emb_all,
            train_pct=TRAIN_PCT,
            test_pct=TEST_PCT,
            val_pct=VAL_PCT,
            random_state=RANDOM_STATE,
        )

    print(f"\nData split: Train={len(splits['X_train'])}, Val={len(splits['X_val'])}, Test={len(splits['X_test'])}")

    print("\n" + "=" * 50)
    print("STARTING MODEL TRAINING")
    print("=" * 50)

    model = train_xgboost_model(splits, params=custom_params)
    test_predictions = evaluate_on_test(model, splits)

    # Feature importance
    feature_names = list(splits['X_train'].columns)
    if splits.get('desc_embeddings_train') is not None:
        n_desc = splits['desc_embeddings_train'].shape[1]
        feature_names.extend([f'desc_emb_{i}' for i in range(n_desc)])
    if splits.get('img_embeddings_train') is not None:
        n_img = splits['img_embeddings_train'].shape[1]
        feature_names.extend([f'img_emb_{i}' for i in range(n_img)])

    if hasattr(model, 'feature_importances_'):
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        print("\n" + "=" * 50)
        print("TOP 10 MOST IMPORTANT FEATURES:")
        print("=" * 50)
        print(fi_df.to_string(index=False))
    
    save_feature_metadata(splits, data)
    save_input_schema(splits)
    save_model_report(model, splits)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print("\nFiles created:")
    print("  ✓ xgboost_model.json")
    print("  ✓ model_metadata.json")
    print("  ✓ api_input_schema.json")
    print("  ✓ model_report.json")
    print("  ✓ test_predictions.csv")