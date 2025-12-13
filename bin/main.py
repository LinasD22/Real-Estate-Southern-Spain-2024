import streamlit as st
import pandas as pd
import os
import re

st.set_page_config(
    page_title="Real Estate Southern Spain 2024",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Real Estate Southern Spain 2024")
st.markdown("---")


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


df = load_data()

if df is None:
    st.error("Unable to load the properties data. Please ensure the properties.csv file exists in the data folder.")
    st.stop()


# ---------- Sidebar filters ----------
st.sidebar.header("🔎 Filters")

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

sel_price = st.sidebar.slider(
    "Price (€)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=5000 if max_price - min_price >= 50000 else 1000
)

# Feature multiselect (flatten)
all_features = sorted({f for fs in df["features_list"] for f in fs})
sel_features = st.sidebar.multiselect("Must include features", options=all_features, default=[])


# Apply filters
fdf = df.copy()

if sel_locations and "location" in fdf.columns:
    fdf = fdf[fdf["location"].isin(sel_locations)]

if sel_types and "property_type" in fdf.columns:
    fdf = fdf[fdf["property_type"].isin(sel_types)]

if "bedrooms" in fdf.columns:
    fdf = fdf[fdf["bedrooms"].fillna(-1) >= sel_beds]

if "bathrooms" in fdf.columns:
    fdf = fdf[fdf["bathrooms"].fillna(-1) >= sel_baths]

if "price_eur" in fdf.columns:
    fdf = fdf[fdf["price_eur"].between(sel_price[0], sel_price[1], inclusive="both")]

if sel_features:
    fdf = fdf[fdf["features_list"].apply(lambda fs: all(feat in fs for feat in sel_features))]


# ---------- Overview ----------
st.subheader("📊 Dataset Overview")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Filtered Properties", len(fdf))
with c2:
    avg_price = fdf["price_eur"].mean() if fdf["price_eur"].notna().any() else None
    st.metric("Avg Price", f"€{avg_price:,.0f}" if avg_price else "—")
with c3:
    st.metric("Locations", fdf["location"].nunique() if "location" in fdf.columns else 0)
with c4:
    st.metric("Property Types", fdf["property_type"].nunique() if "property_type" in fdf.columns else 0)

st.markdown("---")


# ---------- Selection ----------
st.subheader("🔍 Select a Property")

if len(fdf) == 0:
    st.info("No properties match your filters.")
    st.stop()

# Build display strings from filtered df
options = list(fdf.index)

def fmt_row(idx):
    r = fdf.loc[idx]
    pid = r.get("id", idx)
    ptype = r.get("property_type", "Property")
    loc = r.get("location", "Unknown location")
    price = r.get("price_eur", None)
    beds = r.get("bedrooms", None)
    baths = r.get("bathrooms", None)
    price_str = f"€{int(price):,}" if pd.notna(price) else "€—"
    bb = f"{int(beds)} bed" if pd.notna(beds) else "— bed"
    ba = f"{int(baths)} bath" if pd.notna(baths) else "— bath"
    return f"ID {pid}: {ptype} in {loc} - {price_str} ({bb}, {ba})"

selected_idx = st.selectbox(
    "Choose a property to view details:",
    options=options,
    format_func=fmt_row
)

st.markdown("---")
st.subheader("🏡 Property Details")

row = fdf.loc[selected_idx]

col1, col2 = st.columns(2)
with col1:
    st.write("**Basic Information**")
    st.write(f"**ID:** {row.get('id', '—')}")
    st.write(f"**Location:** {row.get('location', '—')}")
    st.write(f"**Property Type:** {row.get('property_type', row.get('title', '—'))}")
    price = row.get("price_eur", pd.NA)
    st.write(f"**Price:** {'€' + format(int(price), ',') if pd.notna(price) else '—'}")
    indoor = row.get('indoor_sqm', pd.NA)
    outdoor = row.get('outdoor_sqm', pd.NA)
    st.write(f"**Indoor Area:** {int(indoor) if pd.notna(indoor) else '—'} m²")
    st.write(f"**Outdoor Area:** {int(outdoor) if pd.notna(outdoor) else '—'} m²")

with col2:
    st.write("**Rooms**")
    st.write(f"**Bedrooms:** {row.get('bedrooms', '—')}")
    st.write(f"**Bathrooms:** {row.get('bathrooms', '—')}")
    st.write("**Features (tags)**")
    feats = row.get("features_list", [])
    if feats:
        st.caption(" | ".join(feats))
    else:
        st.caption("—")

with st.expander("📋 View Raw Data"):
    st.json(row.to_dict())

st.markdown("---")
st.subheader("📋 Filtered Dataset")
show_cols = [c for c in ["id", "location", "property_type", "price_eur", "bedrooms", "bathrooms", "indoor_sqm", "outdoor_sqm", "features"] if c in fdf.columns]
st.dataframe(fdf[show_cols] if show_cols else fdf, use_container_width=True)