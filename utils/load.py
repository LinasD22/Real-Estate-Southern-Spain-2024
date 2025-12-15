import streamlit as st
import pandas as pd
import os
import re
from pathlib import Path

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

def get_property_images(property_id):
    """
    Get all image paths for a property.
    Images should be in data/images/{property_id}/ folder.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    images_dir = os.path.join(parent_dir, "data", "images", str(property_id))

    if not os.path.exists(images_dir):
        return []

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    # Get all image files
    image_files = []
    for file in os.listdir(images_dir):
        if os.path.splitext(file)[1].lower() in image_extensions:
            full_path = os.path.join(images_dir, file)
            image_files.append(full_path)

    return sorted(image_files)


@st.cache_data
def load_test_data():
    """Load only the test dataset properties by recreating the train/test split."""
    from sklearn.model_selection import train_test_split
    
    # Load all properties
    df_all = load_data()
    if df_all is None:
        return None
    
    # Use the same random_state as in preproc.py (42)
    # First split: 80% train+val, 20% test
    train_val_df, test_df = train_test_split(
        df_all, 
        test_size=0.2, 
        random_state=42
    )
    
    # Return test dataset
    return test_df.reset_index(drop=True)


def display_property_slideshow(property_id, container=None):
    """
    Display a slideshow of property images using Streamlit.

    Args:
        property_id: The property ID to fetch images for
        container: Optional Streamlit container to display in
    """
    images = get_property_images(property_id)

    if not images:
        if container:
            container.info(f"No images available for this property.")
        else:
            st.info(f"No images available for this property.")
        return

    # Create a container for the slideshow
    if container is None:
        container = st

    # Initialize slideshow index in session state
    slideshow_key = f"slideshow_{property_id}"
    if slideshow_key not in st.session_state:
        st.session_state[slideshow_key] = 0

    # Display current image and controls
    col1, col2, col3 = container.columns([1, 8, 1])

    with col1:
        if st.button("Previous", key=f"prev_{property_id}"):
            st.session_state[slideshow_key] = (st.session_state[slideshow_key] - 1) % len(images)
            st.rerun()

    with col2:
        current_image_path = images[st.session_state[slideshow_key]]
        st.image(current_image_path, use_container_width=True)
        image_name = os.path.basename(current_image_path)
        st.caption(f"Image {st.session_state[slideshow_key] + 1} of {len(images)}: {image_name}")

    with col3:
        if st.button("Next", key=f"next_{property_id}"):
            st.session_state[slideshow_key] = (st.session_state[slideshow_key] + 1) % len(images)
            st.rerun()

    # Image selector
    image_names = [os.path.basename(img) for img in images]
    selected_image_idx = st.select_slider(
        "Select image",
        options=list(range(len(images))),
        value=st.session_state[slideshow_key],
        key=f"slider_{property_id}"
    )

    if selected_image_idx != st.session_state[slideshow_key]:
        st.session_state[slideshow_key] = selected_image_idx
        st.rerun()
