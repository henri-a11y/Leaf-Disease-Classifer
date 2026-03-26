from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.preprocessing import preprocess_for_mobilenet_v2
from utils.segmentation import HSVGreenMaskConfig, ensure_uint8_rgb, segment_leaf_hsv


def _save_class_indices(class_indices: dict[str, int], output_path: Path) -> None:
    """
    Save mapping for inference.

    ImageDataGenerator gives class_indices as {class_name: index}.
    For app usage we often want the inverse.
    """
    inverse = {str(index): name for name, index in class_indices.items()}
    output_path.write_text(json.dumps(inverse, indent=2), encoding="utf-8")


def build_model(num_classes: int, learning_rate: float = 1e-4) -> Model:
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False  # Freeze base layers (transfer learning)

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="leaf_disease_classifier")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_preprocessing_function(
    *,
    use_segmentation: bool,
    coverage_threshold: float,
    green_mask_config: HSVGreenMaskConfig,
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """
    Returns a preprocessing_function compatible with ImageDataGenerator.

    ImageDataGenerator passes a single image array; we return an image array
    in the [-1, 1] range expected by MobileNetV2 weights.
    """
    if not use_segmentation:
        def mobilenet_preprocess(img: np.ndarray) -> np.ndarray:
            img_u8 = ensure_uint8_rgb(img)
            return preprocess_for_mobilenet_v2(img_u8)

        return mobilenet_preprocess

    def segmentation_preprocess(img: np.ndarray) -> np.ndarray:
        img_u8 = ensure_uint8_rgb(img)
        segmented, _mask, coverage = segment_leaf_hsv(
            img_u8,
            config=green_mask_config,
            coverage_threshold=coverage_threshold,
            raise_on_low_coverage=False,  # fallback instead of failing training
        )

        # If segmentation fails to find a leaf, fall back to original image.
        if coverage < coverage_threshold:
            segmented = img_u8

        return preprocess_for_mobilenet_v2(segmented)

    return segmentation_preprocess


def main() -> None:
    # Ensure local imports work even if run from a different working directory.
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(
        description="Train an Agricultural Leaf Disease Classifier with MobileNetV2."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Root dataset directory (PlantVillage-style: dataset/<class_name>/image.jpg).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save the trained model and class indices.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_segmentation",
        action="store_true",
        help="If set, apply HSV segmentation to images during training preprocessing (default: enabled).",
    )
    parser.add_argument(
        "--no_segmentation",
        action="store_true",
        help="Disable HSV segmentation preprocessing during training.",
    )
    parser.add_argument(
        "--seg_coverage_threshold",
        type=float,
        default=0.01,
        help="Minimum mask coverage ratio to consider an image as a valid leaf.",
    )
    args = parser.parse_args()

    if args.use_segmentation and args.no_segmentation:
        raise ValueError("Use only one of --use_segmentation or --no_segmentation.")

    # By default, segmentation preprocessing is enabled to match the inference pipeline.
    use_segmentation = not args.no_segmentation

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            f"Expected structure like dataset/<class_name>/image.jpg"
        )

    tf.keras.utils.set_random_seed(args.seed)

    preprocessing_function = get_preprocessing_function(
        use_segmentation=use_segmentation,
        coverage_threshold=args.seg_coverage_threshold,
        green_mask_config=HSVGreenMaskConfig(),
    )

    datagen = ImageDataGenerator(
        validation_split=args.val_split,
        preprocessing_function=preprocessing_function,
    )

    train_gen = datagen.flow_from_directory(
        str(dataset_dir),
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=args.seed,
    )

    val_gen = datagen.flow_from_directory(
        str(dataset_dir),
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=args.seed,
    )

    num_classes = len(train_gen.class_indices)
    if num_classes < 2:
        raise ValueError(f"Need at least 2 classes for classification. Found: {num_classes}")

    _save_class_indices(train_gen.class_indices, output_dir / "class_indices.json")

    model = build_model(num_classes=num_classes, learning_rate=args.learning_rate)

    callbacks = [
        ModelCheckpoint(
            filepath=str(output_dir / "leaf_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Ensure a final saved model exists even if callbacks don't save for some reason.
    final_path = output_dir / "leaf_model.h5"
    if not final_path.exists():
        model.save(str(final_path))

    print("Training complete.")
    print(f"Saved model to: {final_path}")
    print(f"Saved class mapping to: {output_dir / 'class_indices.json'}")


if __name__ == "__main__":
    main()

