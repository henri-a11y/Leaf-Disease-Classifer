# Agricultural Leaf Disease Classifier with Segmentation and Classification

This project:
1. Segments a plant leaf using HSV green masking and morphological cleanup.
2. Classifies disease from the segmented leaf using MobileNetV2 transfer learning.
3. Supports both command-line inference and an optional Streamlit UI.

## Features

- Image preprocessing: resize, normalize, optional blur
- HSV leaf segmentation with `cv2.inRange`
- Morphological mask cleanup (closing + opening)
- MobileNetV2-based disease classification
- CLI prediction output:
  - predicted class
  - confidence score
  - saved segmented mask image
- Optional Streamlit app for interactive usage

## Dataset Format (PlantVillage-style)

Dataset is not included due to size limits.

Download: [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

Expected structure:

```text
leaf-disease-project/
  dataset/
    class_name_1/
      img1.jpg
      img2.jpg
    class_name_2/
      img3.jpg
      img4.jpg
```

`train.py` uses `ImageDataGenerator.flow_from_directory()`, so each class folder name becomes the label.

## Setup

Run in terminal from project root:

```bash
pip install -r requirements.txt
```

## Full Command-Line Workflow (No GUI Required)

### 1) Train model

```bash
python train.py --dataset_dir dataset --epochs 10 --batch_size 32
```

Training outputs:
- `models/leaf_model.h5`
- `models/class_indices.json`

### 2) Predict from terminal

```bash
python predict.py --image "path/to/leaf_image.jpg"
```

Prediction outputs:
- Predicted disease class (terminal)
- Confidence score (terminal)
- Segmentation mask at `outputs/mask.png`
- Segmented leaf image at `outputs/segmented.png`

Optional custom output paths:

```bash
python predict.py --image "path/to/leaf_image.jpg" --output_mask "outputs/my_mask.png" --output_segmented "outputs/my_segmented.png"
```

## Optional Streamlit UI

```bash
python -m streamlit run app.py
```

## Technologies Used

- Python
- OpenCV
- NumPy
- TensorFlow / Keras
- Streamlit

