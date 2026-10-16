# Plant Disease Detection

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)

# Potato Leaf Disease Detector

A local web app that classifies potato leaf photos as **Early Blight**, **Late Blight**, or **Healthy**. It uses a transfer-learned MobileNetV2 model, served by FastAPI.

> This is not a general plant-disease detector. It has only been trained on potato leaves, and its confidence score is not a reliable detector for unsupported plants or non-leaf images. Use predictions as a screening aid, not as a substitute for expert diagnosis.

## Dataset and training

The included dataset is expected at `potato!/potato!/` with one folder per class. The current data contains 1,000 early-blight images, 1,000 late-blight images, and 152 healthy images. Training uses deterministic, class-stratified train/validation/test splits, class weights for the smaller healthy class, image augmentation, and ImageNet-pretrained MobileNetV2 weights.

From the repository root, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r api\requirements.txt
```

Train the model (the first run downloads the public MobileNetV2 ImageNet weights):

```powershell
python pot.py
```

This saves `potato_disease_model.h5` and `potato_disease_model.json` in the repository root. The training script prints held-out test accuracy and per-class recall. Keep these metrics with any model you deploy; accuracy varies with the dataset and training run.

## Run the app

After training, start the API from the repository root:

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>, upload a potato leaf photo, and select **Analyse Plant**. JPG, PNG, BMP, and WebP images up to 10 MB are accepted. The API is also available at `POST /predict` with a multipart `file` field. Low-confidence predictions are returned as `Unknown`; this does not guarantee that a confident prediction is a potato leaf.

The API resolves the model and frontend paths from the repository location, so it can also be started from the `api` directory with `uvicorn main:app --reload --port 8000`.

---
