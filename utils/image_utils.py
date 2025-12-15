import streamlit as st
from pathlib import Path
from typing import List, Dict, Optional


def get_dataset_property_images(property_id) -> List[str]:

    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    images_dir = parent_dir / "data" / "images" / str(property_id)

    if not images_dir.exists() or not images_dir.is_dir():
        return []

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    image_files = []
    for file in images_dir.iterdir():
        if file.suffix.lower() in image_extensions:
            image_files.append(file)

    return sorted(image_files)


def get_dataset_images_as_bytes(property_id) -> List[bytes]:

    image_paths = get_dataset_property_images(property_id)
    image_bytes = []

    for img_path in image_paths:
        try:
            with open(img_path, 'rb') as f:
                image_bytes.append(f.read())
        except Exception as e:
            st.warning(f"Could not load image {img_path}: {e}")

    return image_bytes


def load_description_for_property(property_id) -> Optional[str]:
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    desc_file = parent_dir / "data" / "descriptions" / f"{property_id}.txt"

    if desc_file.exists():
        try:
            return desc_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            st.warning(f"Could not load description: {e}")
            return None

    return None


def display_image_gallery(image_data_list: List[Dict]) -> None:
    if not image_data_list:
        st.info("No images uploaded yet.")
        return

    st.subheader(f"Image Gallery ({len(image_data_list)} images)")

    cols = st.columns(3)
    for idx, image_info in enumerate(image_data_list):
        col_idx = idx % 3
        with cols[col_idx]:
            st.image(image_info['data'], caption=image_info['name'], use_container_width=True)
            if st.button("Delete", key=f"delete_img_{idx}"):
                st.session_state.user_property_images.pop(idx)
                st.rerun()


def convert_image_to_bytes(image_path: Path) -> bytes:
    """Convert image file to bytes."""
    with open(image_path, 'rb') as f:
        return f.read()

