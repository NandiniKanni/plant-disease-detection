
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

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
MODEL_PATH = Path(
    os.environ.get("POTATO_MODEL_PATH", ROOT_DIR / "potato_disease_model.h5")
)
METADATA_PATH = MODEL_PATH.with_suffix(".json")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
DEFAULT_CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

app = FastAPI(title="Potato Leaf Disease Detector")
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model(MODEL_PATH)
if METADATA_PATH.exists():
    MODEL_METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    CLASS_NAMES = MODEL_METADATA["class_names"]
    CONFIDENCE_THRESHOLD = float(
        MODEL_METADATA.get("confidence_threshold", 0.7)
    )
else:
    MODEL_METADATA = {}
    CLASS_NAMES = DEFAULT_CLASS_NAMES
    CONFIDENCE_THRESHOLD = 0.7

if MODEL.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        f"Model has {MODEL.output_shape[-1]} outputs but "
        f"{len(CLASS_NAMES)} class names were configured"
    )
IMAGE_HEIGHT, IMAGE_WIDTH = (int(size) for size in MODEL.input_shape[1:3])

@app.get("/")
async def serve_frontend():
    return HTMLResponse((FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))

@app.get("/ping")
async def ping():
    return {"status": "ok", "classes": CLASS_NAMES}

def read_file_as_image(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            resample=Image.Resampling.BILINEAR,
        )
        return np.asarray(image, dtype=np.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_SIZE + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_UPLOAD_SIZE // (1024 * 1024)} MB limit.",
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

    probabilities = MODEL.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "predicted_class": "Unknown",
            "confidence": confidence,
            "message": (
                "Low confidence. This model only classifies potato leaves; "
                "other plants and image types are not supported."
            ),
        }

    return {
        "predicted_class": CLASS_NAMES[class_index],
        "confidence": confidence,
        "message": "Potato leaf prediction. Confirm with an agronomist if unsure.",
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
