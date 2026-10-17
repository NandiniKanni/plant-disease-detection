
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"

# Use the newly trained model by default
MODEL_PATH = Path(
    os.environ.get(
        "POTATO_MODEL_PATH",
        str(ROOT_DIR / "potato_disease_model.keras"),
    )
)
METADATA_PATH = MODEL_PATH.with_suffix(".json")

MAX_UPLOAD_SIZE = 10 * 1024 * 1024

# Load model
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

MODEL = tf.keras.models.load_model(MODEL_PATH)

# Load class names and settings
if METADATA_PATH.is_file():
    MODEL_METADATA = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )
    CLASS_NAMES = MODEL_METADATA["class_names"]
    CONFIDENCE_THRESHOLD = float(
        MODEL_METADATA.get("confidence_threshold", 0.7)
    )
    IMAGE_SIZE = int(MODEL_METADATA.get("image_size", 256))
else:
    raise FileNotFoundError(
        f"Model metadata not found: {METADATA_PATH}"
    )

if MODEL.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        f"Model outputs {MODEL.output_shape[-1]} classes, "
        f"but metadata contains {len(CLASS_NAMES)} class names."
    )

IMAGE_HEIGHT, IMAGE_WIDTH = (
    int(size) for size in MODEL.input_shape[1:3]
)

app = FastAPI(title="Potato Leaf Disease Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.is_dir():
    app.mount(
        "/frontend",
        StaticFiles(directory=FRONTEND_DIR),
        name="frontend",
    )


@app.get("/")
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"

    if not index_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Frontend index.html was not found.",
        )

    return HTMLResponse(
        index_path.read_text(encoding="utf-8")
    )


@app.get("/ping")
async def ping():
    return {
        "status": "ok",
        "model_file": MODEL_PATH.name,
        "classes": CLASS_NAMES,
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
    }


def read_file_as_image(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            resample=Image.Resampling.BILINEAR,
        )

        # Keep values in the 0-255 range.
        # The model performs rescaling internally.
        return np.asarray(image, dtype=np.float32)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_SIZE + 1)

    if not data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit.",
        )

    try:
        image = read_file_as_image(data)
    except (
        Image.DecompressionBombError,
        Image.UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail="Upload a valid JPG, PNG, BMP, or WebP image.",
        ) from error

    predictions = MODEL.predict(
        np.expand_dims(image, axis=0),
        verbose=0,
    )[0]

    class_index = int(np.argmax(predictions))
    confidence_fraction = float(predictions[class_index])
    confidence_percent = round(confidence_fraction, 2)

    actual_class = CLASS_NAMES[class_index]
    is_confident = (
        confidence_fraction >= CONFIDENCE_THRESHOLD
    )

    # Return probabilities as percentages
    all_probabilities = {
        name: round(float(probability) * 100, 2)
        for name, probability in zip(CLASS_NAMES, predictions)
    }

    return {
        "predicted_class": (
            actual_class if is_confident else "Unknown"
        ),
        "class_name": actual_class,
        "confidence": confidence_percent,
        "confidence_fraction": confidence_fraction,
        "is_confident": is_confident,
        "probabilities": all_probabilities,
        "message": (
            "Potato leaf prediction. Confirm with an agronomist if unsure."
            if is_confident
            else (
                "Low confidence. Try a clear potato leaf image "
                "with good lighting."
            )
        ),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
