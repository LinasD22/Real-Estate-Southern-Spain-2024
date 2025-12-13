import streamlit as st
import pandas as pd
import os
import re

def get_df_data(df):
    location_vals = sorted([x for x in df["location"].dropna().unique()]) if "location" in df.columns else []
    type_vals = sorted([x for x in df["property_type"].dropna().unique()]) if "property_type" in df.columns else []

    min_price = int(df["price_eur"].min()) if df["price_eur"].notna().any() else 0
    max_price = int(df["price_eur"].max()) if df["price_eur"].notna().any() else 0

    sel_locations = st.sidebar.multiselect("Location", options=location_vals, default=[])
    sel_types = st.sidebar.multiselect("Property type", options=type_vals, default=[])

    beds_min, beds_max = 0, int(df["bedrooms"].max()) if df["bedrooms"].notna().any() else 0
    baths_min, baths_max = 0, int(df["bathrooms"].max()) if df["bathrooms"].notna().any() else 0

    sel_beds = st.sidebar.slider("Bedrooms (min)", min_value=beds_min, max_value=beds_max, value=beds_min)
    sel_baths = st.sidebar.slider("Bathrooms (min)", min_value=baths_min, max_value=baths_max, value=baths_min)
    return min_price, max_price, sel_locations, sel_types, sel_beds, sel_baths

@st.cache_data
def load_data():
    """Load properties.csv and normalize columns for the provided schema."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    csv_path = os.path.join(parent_dir, "data", "properties.csv")

    try:
        df = pd.read_csv(csv_path)

        # Expected (based on your sample):
        # id, location, price, title, bedrooms, bathrooms, size_sqm, <optional>, features
        # We'll be defensive about exact column names.

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Map common possibilities
        # If your CSV already has these exact names, nothing changes.
        col_map_candidates = {
            "id": ["reference", "id", "property_id", "listing_id"],
            "location": ["location", "area", "town"],
            "price": ["price", "price_text", "price_eur_text"],
            "title": ["title", "property_title", "type_title"],
            "bedrooms": ["bedrooms", "beds", "bedroom"],
            "bathrooms": ["bathrooms", "baths", "bathroom"],
            "indoor_sqm": ["indoor surface area in sqm", "indoor_sqm", "indoor_surface_area", "size_sqm", "size", "built_area"],
            "outdoor_sqm": ["outdoor surface area in sqm", "outdoor_sqm", "outdoor_surface_area", "plot_size"],
            "features": ["pipe-separed list of features of the property", "features", "amenities", "feature_list"]
        }

        def first_existing(cols):
            for c in cols:
                if c in df.columns:
                    return c
            return None

        resolved = {k: first_existing(v) for k, v in col_map_candidates.items()}

        # Rename resolved columns to standard names
        rename_map = {resolved[k]: k for k in resolved if resolved[k] is not None and resolved[k] != k}
        df = df.rename(columns=rename_map)

        # --- Parse price to numeric ---
        if "price_eur" not in df.columns:
            if "price" in df.columns:
                # handles "€450,000" / "450000" / "€ 450.000" etc.
                df["price_eur"] = (
                    df["price"]
                    .astype(str)
                    .str.replace("€", "", regex=False)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                df["price_eur"] = pd.to_numeric(df["price_eur"], errors="coerce")
            else:
                df["price_eur"] = pd.NA

        # --- Parse title like "3 Bedroom Apartment" into bedrooms + property_type ---
        if "property_type" not in df.columns:
            if "title" in df.columns:
                def parse_title(t):
                    t = str(t)
                    m = re.match(r"^\s*(\d+)\s*Bedroom\s+(.*)\s*$", t, flags=re.IGNORECASE)
                    if m:
                        return int(m.group(1)), m.group(2).strip()
                    return None, t.strip()

                parsed = df["title"].apply(parse_title)
                df["bedrooms_from_title"] = parsed.apply(lambda x: x[0])
                df["property_type"] = parsed.apply(lambda x: x[1])
            else:
                df["property_type"] = pd.NA
                df["bedrooms_from_title"] = pd.NA

        # Prefer explicit bedrooms column; if missing/NA use parsed
        if "bedrooms" not in df.columns:
            df["bedrooms"] = df.get("bedrooms_from_title", pd.NA)
        else:
            df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
            if "bedrooms_from_title" in df.columns:
                df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms_from_title"])

        # Bathrooms / size numeric
        if "bathrooms" in df.columns:
            df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")
        else:
            df["bathrooms"] = pd.NA

        if "indoor_sqm" in df.columns:
            df["indoor_sqm"] = pd.to_numeric(df["indoor_sqm"], errors="coerce")
        else:
            df["indoor_sqm"] = pd.NA

        if "outdoor_sqm" in df.columns:
            df["outdoor_sqm"] = pd.to_numeric(df["outdoor_sqm"], errors="coerce")
        else:
            df["outdoor_sqm"] = pd.NA

        # --- Features split (pipe-separated) ---
        if "features" in df.columns:
            df["features_list"] = (
                df["features"]
                .fillna("")
                .astype(str)
                .apply(lambda s: [x.strip() for x in s.split("|") if x.strip()])
            )
        else:
            df["features_list"] = [[] for _ in range(len(df))]

        return df

    except FileNotFoundError:
        st.error(f"Error: Could not find properties.csv at {csv_path}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

