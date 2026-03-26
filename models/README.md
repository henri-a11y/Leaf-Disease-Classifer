# Models folder

This folder is where training outputs are saved.

After running:

```bash
python train.py --dataset_dir dataset
```

you should see:

- `leaf_model.h5`: trained Keras model (MobileNetV2 transfer learning)
- `class_indices.json`: mapping from model output index → class name (used by `app.py`)

If these files are missing, the Streamlit app will prompt you to train first.

