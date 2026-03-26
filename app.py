from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils.preprocessing import preprocess_for_mobilenet_v2
from utils.segmentation import HSVGreenMaskConfig, segment_leaf_hsv


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "leaf_model.h5"
CLASS_INDICES_PATH = PROJECT_ROOT / "models" / "class_indices.json"

# Ensure local imports work even if Streamlit changes the working directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


@st.cache_data(show_spinner=False)
def load_class_names():
    if not CLASS_INDICES_PATH.exists():
        return {}
    data = json.loads(CLASS_INDICES_PATH.read_text(encoding="utf-8"))
    # keys are strings of indices, values are class names
    return data


def predict(model, segmented_rgb: np.ndarray) -> tuple[str, float, np.ndarray]:
    """
    Predict top class.

    Returns
    -------
    class_name: str
    confidence: float (0..1)
    probs: np.ndarray of shape (num_classes,)
    """
    x = preprocess_for_mobilenet_v2(segmented_rgb)
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
    probs = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(probs))
    class_name = st.session_state.get("class_names", {}).get(str(idx), str(idx))
    confidence = float(probs[idx])
    return class_name, confidence, probs


def main() -> None:
    st.set_page_config(page_title="Leaf Disease Classifier", layout="centered")
    st.title("Agricultural Leaf Disease Classifier with Segmentation")
    st.caption("Upload a leaf image; we segment the leaf and classify disease using MobileNetV2.")

    class_names = load_class_names()
    st.session_state["class_names"] = class_names

    model = load_model()
    if model is None:
        st.error(
            "Model file not found. Train the model first by running `python train.py` "
            "from the project folder."
        )
        st.stop()

    uploaded = st.file_uploader(
        "Upload an image of a plant leaf",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        return

    # Read image as RGB for OpenCV's RGB->HSV conversion.
    try:
        img_pil = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Could not read image: {e}")
        return

    image_rgb = np.array(img_pil)
    st.subheader("Input")
    st.image(image_rgb, caption="Original image", use_column_width=True)

    st.subheader("Segmentation")
    green_config = HSVGreenMaskConfig()

    # Segmentation threshold is a heuristic. Tune it if you see too many failures.
    coverage_threshold = st.slider(
        "Leaf mask coverage threshold",
        min_value=0.001,
        max_value=0.2,
        value=0.01,
        step=0.001,
        help="Higher values require more green pixels to consider the image a leaf.",
    )

    try:
        segmented_rgb, mask, coverage = segment_leaf_hsv(
            image_rgb,
            config=green_config,
            coverage_threshold=float(coverage_threshold),
            raise_on_low_coverage=True,
            mask_output_type="uint8",
        )
    except ValueError as e:
        st.error(str(e))
        st.warning(
            "Segmentation indicates this image may not contain a clear leaf. "
            "Try another image with a visible green leaf region."
        )
        return

    st.write(f"Leaf mask coverage: {coverage * 100.0:.2f}%")

    st.image(segmented_rgb, caption="Segmented leaf (background removed)", use_column_width=True)
    st.image(mask, caption="Segmentation mask", use_column_width=True)

    st.subheader("Disease Classification")
    class_name, confidence, _probs = predict(model, segmented_rgb)

    st.success(f"Predicted disease class: `{class_name}`")
    st.metric("Confidence", f"{confidence * 100.0:.2f}%")


if __name__ == "__main__":
    main()

