import streamlit as st
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="Real Estate Southern Spain 2024",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 Real Estate Southern Spain 2024")
st.markdown("---")

# Load the CSV file
@st.cache_data
def load_data():
    """Load the properties.csv file from the data folder."""
    # Get the path to the data folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    csv_path = os.path.join(parent_dir, "data", "properties.csv")
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: Could not find properties.csv at {csv_path}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Display some statistics
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(df))
    with col2:
        st.metric("Avg Price", f"€{df['price_eur'].mean():,.0f}")
    with col3:
        st.metric("Locations", df['location'].nunique())
    with col4:
        st.metric("Property Types", df['property_type'].nunique())
    
    st.markdown("---")
    
    # Dropdown to select a property row
    st.subheader("🔍 Select a Property")
    
    # Create a display string for each property
    property_options = []
    for idx, row in df.iterrows():
        display_str = f"ID {row['id']}: {row['property_type']} in {row['location']} - €{row['price_eur']:,} ({row['bedrooms']} bed, {row['bathrooms']} bath)"
        property_options.append(display_str)
    
    # Dropdown selection
    selected_property = st.selectbox(
        "Choose a property to view details:",
        options=range(len(df)),
        format_func=lambda x: property_options[x]
    )
    
    # Display selected property details
    st.markdown("---")
    st.subheader("🏡 Property Details")
    
    selected_row = df.iloc[selected_property]
    
    # Display property details in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Information**")
        st.write(f"**ID:** {selected_row['id']}")
        st.write(f"**Location:** {selected_row['location']}")
        st.write(f"**Property Type:** {selected_row['property_type']}")
        st.write(f"**Price:** €{selected_row['price_eur']:,}")
        st.write(f"**Year Built:** {selected_row['year_built']}")
    
    with col2:
        st.write("**Property Features**")
        st.write(f"**Bedrooms:** {selected_row['bedrooms']}")
        st.write(f"**Bathrooms:** {selected_row['bathrooms']}")
        st.write(f"**Size:** {selected_row['size_sqm']} m²")
        st.write(f"**Pool:** {selected_row['pool']}")
        st.write(f"**Garage:** {selected_row['garage']}")
        st.write(f"**Distance to Beach:** {selected_row['distance_to_beach_km']} km")
    
    st.markdown("---")
    
    # Show the raw data for the selected property
    with st.expander("📋 View Raw Data"):
        st.json(selected_row.to_dict())
    
    # Show full dataset
    st.markdown("---")
    st.subheader("📋 Full Dataset")
    st.dataframe(df, use_container_width=True)

else:
    st.error("Unable to load the properties data. Please ensure the properties.csv file exists in the data folder.")
