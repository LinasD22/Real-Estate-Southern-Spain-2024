import streamlit as st
import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from utils import load_data, get_df_data

st.set_page_config(
    page_title="Real Estate Southern Spain 2024",
    page_icon="🏠",
    layout="wide"
)
st.markdown("---")
st.title("Real Estate Southern Spain 2024")


tab1, tab2 = st.tabs(["Select from dataset", "Predict raw"])





df = load_data()

if df is None:
    st.error("Unable to load the properties data. Please ensure the properties.csv file exists in the data folder.")
    st.stop()


with tab1:
    # ---------- Sidebar filters ----------
    st.sidebar.header("Filters")

    min_price, max_price, sel_locations, sel_types, sel_beds, sel_baths = get_df_data(df)


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
    st.subheader("Dataset Overview")
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
    st.subheader("Property Details")


    row = fdf.loc[selected_idx]
    st.session_state["row_panda_from_df"] = row


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
        # Convert numpy types to Python native types for JSON serialization
        row_dict = {}
        for k, v in row.to_dict().items():
            if isinstance(v, (np.integer, np.floating)):
                row_dict[k] = v.item()
            elif isinstance(v, np.ndarray):
                row_dict[k] = v.tolist()
            else:
                row_dict[k] = v
        st.json(row_dict)

    st.markdown("---")
    st.subheader("Filtered Dataset")
    show_cols = [c for c in ["id", "location", "property_type", "price_eur", "bedrooms", "bathrooms", "indoor_sqm", "outdoor_sqm", "features"] if c in fdf.columns]
    display_df = fdf[show_cols].copy() if show_cols else fdf.copy()
    # Convert all columns to Arrow-compatible types
    for col in display_df.columns:
        if display_df[col].dtype == 'object':
            # Convert object columns to string
            display_df[col] = display_df[col].apply(lambda x: str(x) if x is not None else "")
        elif display_df[col].dtype == 'float64':
            # Ensure float columns don't have object values mixed in
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
    st.dataframe(display_df, width="stretch")

with tab2:
    st.subheader("Enter Property Details")

    with st.form("property_form"):
        col1, col2 = st.columns(2)

        with col1:
            input_location = st.text_input("Location")
            input_property_type = st.selectbox(
                "Property Type",
                options=df["property_type"].dropna().unique().tolist() if "property_type" in df.columns else []
            )
            input_price = st.number_input("Price € (for metrics)", min_value=0, step=1000)
            input_indoor_sqm = st.number_input("Indoor Area (m²)", min_value=0, step=1)
            input_outdoor_sqm = st.number_input("Outdoor Area (m²)", min_value=0, step=1)

        with col2:
            input_bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
            input_bathrooms = st.number_input("Bathrooms", min_value=0, step=1)
            all_features = sorted(
                {f for fs in df["features_list"] for f in fs}) if "features_list" in df.columns else []
            input_features = st.multiselect("Features", options=all_features)

        submitted = st.form_submit_button("Save Property")

        if submitted:
            new_property = {
                "location": input_location,
                "property_type": input_property_type,
                "price_eur": input_price,
                "indoor_sqm": input_indoor_sqm,
                "outdoor_sqm": input_outdoor_sqm,
                "bedrooms": input_bedrooms,
                "bathrooms": input_bathrooms,
                "features_list": input_features
            }

            # Override - store only one property
            st.session_state.user_property = new_property
            st.success("Property saved!")

    # Display saved property
    if "user_property" in st.session_state and st.session_state.user_property:
        st.markdown("---")
        st.subheader("Saved Property")
        prop = st.session_state.user_property
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Location:** {prop['location']}")
            st.write(f"**Property Type:** {prop['property_type']}")
            st.write(f"**Price:** €{prop['price_eur']:,}")
            st.write(f"**Indoor Area:** {prop['indoor_sqm']} m²")
            st.write(f"**Outdoor Area:** {prop['outdoor_sqm']} m²")
        with col2:
            st.write(f"**Bedrooms:** {prop['bedrooms']}")
            st.write(f"**Bathrooms:** {prop['bathrooms']}")
            st.write(f"**Features:** {', '.join(prop['features_list']) if prop['features_list'] else '—'}")




    st.markdown("---")
    st.subheader("Predict Price")
    st.subheader(f"**Current selected property:**")
    # Check which data sources are available
    has_user_property = "user_property" in st.session_state and st.session_state.user_property
    has_df_property = st.session_state.get("row_panda_from_df") is not None

    if has_user_property and has_df_property:
        st.markdown("---")
        st.info("Both data sources are available. Select which one to use for prediction.")
        prediction_source = st.radio(
            "Select property source:",
            options=["Manual Input (Form)", "Dataset Selection (Tab 1)"],
            horizontal=True
        )
        if prediction_source == "Manual Input (Form)":
            selected_dict = st.session_state.user_property
        else:
            selected_dict = st.session_state.row_panda_from_df.to_dict()
    elif has_user_property:
        selected_dict = st.session_state.user_property
    elif has_df_property:
        selected_dict = st.session_state.row_panda_from_df.to_dict()
    else:
        selected_dict = None

    # Convert to numpy array for prediction
    if selected_dict:
        # Get all feature keys from df columns, excluding non-numeric and putting price_eur last
        exclude_cols = ["id", "title", "price", "features", "features_list", "bedrooms_from_title", "location", "property_type"]
        numeric_cols = [col for col in df.columns if col not in exclude_cols and col != "price_eur"]
        feature_keys = numeric_cols + ["price_eur"]  # price_eur as last column

        # Build array with None for missing values
        property_values = []
        for key in feature_keys:
            val = selected_dict.get(key, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                property_values.append(np.nan)
            else:
                property_values.append(val)

        property_array = np.array([property_values], dtype=np.float64)
        st.session_state.selected_for_prediction = property_array

        # Display as a table
        st.write("**Selected Property Features:**")
        display_table = pd.DataFrame([property_values], columns=feature_keys)
        st.dataframe(display_table, width="stretch")

        # Display property features (tags) after the table
        features_list = selected_dict.get("features_list", [])
        if features_list:
            st.write("**Property Features (Tags):**")
            # Create styled tags using HTML
            tags_html = " ".join([
                f'<span style="background-color: #123D35; padding: 4px 10px; margin: 3px; border-radius: 15px; display: inline-block; font-size: 14px;">{feat}</span>'
                for feat in features_list
            ])
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.write("**Property Features (Tags):** —")

    make_prediction = st.button("Predict property price")

    st.markdown("---")
    st.subheader("Accuracy metrics")

    st.write("Predicted")

    st.write("Actual")

    st.write("Error")
