import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def _apply_caps(df: pd.DataFrame, limit: Dict[str, float]) -> pd.DataFrame:
	df = df.copy()
	for col, cap in limit.items():
		if col in df.columns:
			df[col] = df[col].clip(upper=cap)
	return df


def get_preprocessed_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
	models_dir = Path(__file__).parent
	csv_path = (models_dir.parent / "data" / "properties.csv").resolve()

	df = pd.read_csv(csv_path)
	df = df.copy()
	df.columns = df.columns.str.strip()

	# Price cleanup and target
	df["price"] = (
		df["price"].astype(str).str.replace(r"[^\d.,]", "", regex=True).str.replace(",", "").astype(float)
	)

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
		"Plot",
		"Land",
		"Commercial Property",
		"Commercial Development",
		"Warehouse",
		"Shop",
		"Office",
		"Restaurant",
		"Bar / Nightclub",
		"Cafe",
		"Parking",
		"Garage",
		"Storage",
		"Property",
		"Other",
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
	tr = tr.copy()
	tr["y"] = np.log(tr["price"]) if "price" in tr.columns else 0.0
	global_mean = tr["y"].mean()
	city_stats = tr.groupby("city")["y"].agg(["mean", "count"])
	k = 20
	city_stats["mean"] = (city_stats["count"] * city_stats["mean"] + k * global_mean) / (
		city_stats["count"] + k
	)

	def add_city_mean(df_: pd.DataFrame) -> pd.DataFrame:
		return df_.assign(city_mean=df_["city"].map(city_stats["mean"]).fillna(global_mean))

	tr = add_city_mean(tr)
	val = add_city_mean(val)
	test_df = add_city_mean(test_df)

	# One-hot encode property_type
	ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	ohe.fit(tr[["property_type"]])
	ohe_cols = list(ohe.get_feature_names_out(["property_type"]))

	def apply_ohe(df_: pd.DataFrame) -> pd.DataFrame:
		arr = ohe.transform(df_[["property_type"]])
		df_[ohe_cols] = arr
		return df_

	tr = apply_ohe(tr)
	val = apply_ohe(val)
	test_df = apply_ohe(test_df)

	# Top features flags from features list
	counts = Counter(pd.Series(sum(tr["features"].tolist(), [])))
	top = counts.most_common(30)
	if ("Other", 1) not in top:
		top.append(("Other", 1))
	feature_names = [name for name, _ in top]

	def add_feature_cols(df_: pd.DataFrame, features: List[str]) -> pd.DataFrame:
		for f in features:
			df_[f] = df_["features"].apply(lambda x: int(f in x) if isinstance(x, list) else 0)
		return df_

	tr = add_feature_cols(tr, feature_names)
	val = add_feature_cols(val, feature_names)
	test_df = add_feature_cols(test_df, feature_names)

	# Log-transformed numeric features and targets
	for col in ["indoor surface area in sqm", "outdoor surface area in sqm"]:
		if col in tr.columns:
			tr[f"{col}_log"] = np.log1p(tr[col])
			val[f"{col}_log"] = np.log1p(val[col])
			test_df[f"{col}_log"] = np.log1p(test_df[col])

	# Targets
	y_tr = np.log1p(tr["price"]) if "price" in tr.columns else pd.Series(np.zeros(len(tr)))
	y_val = np.log1p(val["price"]) if "price" in val.columns else pd.Series(np.zeros(len(val)))

	# Final feature columns selection
	base_cols = [c for c in ["city_mean", "bedrooms", "bathrooms", "indoor surface area in sqm", "outdoor surface area in sqm"] if c in tr.columns]
	feature_cols = base_cols + feature_names + ohe_cols

	X_tr = tr[feature_cols].copy()
	X_val = val[feature_cols].copy()
	X_test = test_df[feature_cols].copy()

	return X_tr, y_tr, X_val, y_val, X_test, feature_cols

