from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from utils.preprocessing import preprocess_for_mobilenet_v2
from utils.segmentation import HSVGreenMaskConfig, segment_leaf_hsv


def load_class_indices(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Class index file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI inference for agricultural leaf disease classification."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input leaf image.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/leaf_model.h5",
        help="Path to trained Keras model file.",
    )
    parser.add_argument(
        "--class_indices_path",
        type=str,
        default="models/class_indices.json",
        help="Path to class index JSON file.",
    )
    parser.add_argument(
        "--output_mask",
        type=str,
        default="outputs/mask.png",
        help="Path to save segmented mask image.",
    )
    parser.add_argument(
        "--output_segmented",
        type=str,
        default="outputs/segmented.png",
        help="Path to save segmented RGB image.",
    )
    parser.add_argument(
        "--coverage_threshold",
        type=float,
        default=0.01,
        help="Minimum segmentation coverage threshold.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model_path)
    class_indices_path = Path(args.class_indices_path)
    output_mask = Path(args.output_mask)
    output_segmented = Path(args.output_segmented)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first with: python train.py --dataset_dir dataset"
        )

    class_indices = load_class_indices(class_indices_path)
    model = tf.keras.models.load_model(str(model_path), compile=False)

    image_rgb = read_image_rgb(image_path)
    segmented_rgb, mask, coverage = segment_leaf_hsv(
        image_rgb,
        config=HSVGreenMaskConfig(),
        coverage_threshold=args.coverage_threshold,
        raise_on_low_coverage=True,
        mask_output_type="uint8",
    )

    x = preprocess_for_mobilenet_v2(segmented_rgb)
    x = np.expand_dims(x, axis=0)
    probs = model.predict(x, verbose=0)[0]

    top_idx = int(np.argmax(probs))
    top_class = class_indices.get(str(top_idx), str(top_idx))
    top_conf = float(probs[top_idx])

    # Save outputs
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    output_segmented.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_mask), mask)
    cv2.imwrite(str(output_segmented), cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR))

    print(f"Predicted disease class: {top_class}")
    print(f"Confidence score: {top_conf:.4f} ({top_conf * 100:.2f}%)")
    print(f"Leaf mask coverage: {coverage * 100:.2f}%")
    print(f"Saved mask image: {output_mask}")
    print(f"Saved segmented image: {output_segmented}")


if __name__ == "__main__":
    main()

