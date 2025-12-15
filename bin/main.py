import streamlit as st
import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from utils.load import load_data, load_test_data, get_df_data, display_property_slideshow
from utils.image_utils import get_dataset_images_as_bytes, load_description_for_property
from utils.predict import make_prediction, get_prediction_with_metadata

st.set_page_config(
    page_title="Real Estate Southern Spain 2024",
    page_icon="house",
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
    df_test = load_test_data()
    if df_test is None:
        st.error("Unable to load the test dataset.")
        st.stop()
    
    # ---------- Sidebar filters ----------
    st.sidebar.header("Filters")
    
    # Initialize search mode in session state
    if "search_by_id" not in st.session_state:
        st.session_state.search_by_id = False

    min_price, max_price, sel_locations, sel_types, sel_beds, sel_baths = get_df_data(df_test)

    # Property ID search button
    search_button_label = "🔍 Search by Property ID" if not st.session_state.search_by_id else "Cancel ID Search"
    if st.sidebar.button(search_button_label, key="search_by_id_button"):
        st.session_state.search_by_id = not st.session_state.search_by_id
        st.rerun()

    # Property ID input (only shown when search mode is active)
    property_id_query = None
    if st.session_state.search_by_id:
        property_id_query = st.sidebar.text_input(
            label="Property ID", 
            placeholder="Enter Property ID", 
            key="property_id_search"
        )

    # Regular filters
    if not st.session_state.search_by_id:
        sel_price = st.sidebar.slider(
            "Price (€)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=5000 if max_price - min_price >= 50000 else 1000
        )

        # Feature multiselect (flatten)
        all_features = sorted({f for fs in df_test["features_list"] for f in fs})
        sel_features = st.sidebar.multiselect("Must include features", options=all_features, default=[])

    # Apply filters
    fdf = df_test.copy()
    
    if st.session_state.search_by_id:
        if property_id_query and str(property_id_query).strip():
            id_col = "id" if "id" in fdf.columns else "reference" if "reference" in fdf.columns else None
            if id_col:
                fdf[id_col] = fdf[id_col].astype(str)
                fdf = fdf[fdf[id_col].str.contains(str(property_id_query).strip(), case=False, na=False)]
            else:
                st.warning("ID column not found in dataset.")
                fdf = pd.DataFrame()  # Empty dataframe if no ID column
    else:
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
    st.subheader("Select a Property")

    if len(fdf) == 0:
        st.info("No properties match your filters.")
        st.stop()

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

    with st.expander("View Raw Data"):
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
    st.subheader("Property Images")

    property_id = row.get("id", None)
    if property_id:
        display_property_slideshow(property_id)
    else:
        st.warning("Property ID not found. Unable to load images.")

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
    # st.subheader("Enter Property Details")
    #
    # with st.form("property_form"):
    #     col1, col2 = st.columns(2)
    #
    #     with col1:
    #         input_location = st.text_input("Location")
    #         input_property_type = st.selectbox(
    #             "Property Type",
    #             options=df["property_type"].dropna().unique().tolist() if "property_type" in df.columns else []
    #         )
    #         input_price = st.number_input("Price € (for metrics)", min_value=0, step=1000)
    #         input_indoor_sqm = st.number_input("Indoor Area (m²)", min_value=0, step=1)
    #         input_outdoor_sqm = st.number_input("Outdoor Area (m²)", min_value=0, step=1)
    #
    #     with col2:
    #         input_bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
    #         input_bathrooms = st.number_input("Bathrooms", min_value=0, step=1)
    #         all_features = sorted(
    #             {f for fs in df["features_list"] for f in fs}) if "features_list" in df.columns else []
    #         input_features = st.multiselect("Features", options=all_features)
    #
    #         st.markdown("**Property Images**")
    #         input_images = st.file_uploader(
    #             "Upload property images",
    #             type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
    #             accept_multiple_files=True,
    #             key="form_property_images"
    #         )
    #
    #     submitted = st.form_submit_button("Save Property")
    #
    #     if submitted:
    #         new_property = {
    #             "location": input_location,
    #             "property_type": input_property_type,
    #             "price_eur": input_price,
    #             "indoor_sqm": input_indoor_sqm,
    #             "outdoor_sqm": input_outdoor_sqm,
    #             "bedrooms": input_bedrooms,
    #             "bathrooms": input_bathrooms,
    #             "features_list": input_features
    #         }
    #
    #         # Override - store only one property
    #         st.session_state.user_property = new_property
    #
    #         # Process and save images
    #         if input_images:
    #             st.session_state.user_property_images = []
    #             for uploaded_file in input_images:
    #                 st.session_state.user_property_images.append({
    #                     'name': uploaded_file.name,
    #                     'data': uploaded_file.getvalue(),
    #                     'type': uploaded_file.type
    #                 })
    #         else:
    #             st.session_state.user_property_images = []
    #
    #         st.success("Property saved!")

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

        # Display uploaded images gallery
        if "user_property_images" in st.session_state and st.session_state.user_property_images:
            st.markdown("---")
            st.subheader(f"Uploaded Images ({len(st.session_state.user_property_images)})")

            # Create columns for image gallery
            cols = st.columns(3)
            for idx, image_info in enumerate(st.session_state.user_property_images):
                col_idx = idx % 3
                with cols[col_idx]:
                    st.image(image_info['data'], caption=image_info['name'], width="stretch")
                    if st.button("Delete", key=f"delete_img_{idx}"):
                        st.session_state.user_property_images.pop(idx)
                        st.rerun()

            # Clear all images button
            if st.button("Clear All Images"):
                st.session_state.user_property_images = []
                st.rerun()



    st.markdown("---")
    st.subheader("Predict Price")

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
            use_user_images = True
        else:
            selected_dict = st.session_state.row_panda_from_df.to_dict()
            use_user_images = False
    elif has_user_property:
        selected_dict = st.session_state.user_property
        use_user_images = True
    elif has_df_property:
        selected_dict = st.session_state.row_panda_from_df.to_dict()
        use_user_images = False
    else:
        selected_dict = None
        use_user_images = False

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


    # ---------------------------------- Predict ----------------------------------
    col_pred1, col_pred2 = st.columns(2)

    with col_pred1:
        make_prediction_btn = st.button("Predict property price", key="predict_btn")

    with col_pred2:
        st.write("")

    if make_prediction_btn and selected_dict:
        with st.spinner("Encoding images and analyzing property..."):
            try:
                image_data = None
                if use_user_images and "user_property_images" in st.session_state:
                    image_data = [
                        img_info['data'] for img_info in st.session_state.user_property_images
                    ]
                elif not use_user_images:
                    property_id = selected_dict.get("id")
                    if property_id:
                        image_data = get_dataset_images_as_bytes(property_id)
                        if image_data:
                            st.info(f"Loaded {len(image_data)} images from dataset")

                description = selected_dict.get("description", None)
                if description is None and not use_user_images:
                    property_id = selected_dict.get("id")
                    if property_id:
                        description = load_description_for_property(property_id)
                        if description:
                            st.info("Loaded description from dataset")

                # Make prediction with metadata
                result = get_prediction_with_metadata(
                    selected_dict,
                    image_data_list=image_data,
                    description_text=description
                )

                if result:
                    st.success("Prediction complete!")

                    st.markdown("---")
                    st.subheader("Prediction Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Predicted Price",
                            f"€{result['predicted_price']:,.0f}"
                        )
                    with col2:
                        st.subheader("Actual Price")
                        actual_price = selected_dict.get("price_eur")
                        if actual_price is not None and pd.notna(actual_price):
                            st.metric(
                                "Actual Price",
                                f"€{float(actual_price):,.0f}"
                            )
                        else:
                            st.info("No actual price available for this property.")

                    # Additional info
                    # st.markdown("**Prediction Details:**")
                    # details = {
                    #     "Images provided": "Yes" if result['images_provided'] else "No",
                    #     "Description provided": "Yes" if result['description_provided'] else "No",
                    #     "Confidence range": f"±€{result['predicted_price'] * 0.15:,.0f}"
                    # }
                    #
                    # for key, value in details.items():
                    #     st.write(f"- **{key}:** {value}")

                    st.session_state.last_prediction = result
                else:
                    st.error("Prediction failed. Please check your input data.")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

    # if "last_prediction" in st.session_state:
    #     st.markdown("---")
    #     st.subheader("Prediction")
    #     last = st.session_state.last_prediction
    #     st.info(
    #         f"**€{last['predicted_price']:,.0f}** "
    #         f"(range: €{last['confidence_lower']:,.0f} - €{last['confidence_upper']:,.0f})"
    #     )



