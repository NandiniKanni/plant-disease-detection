
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageOps

# Project folder
ROOT_DIR = Path(__file__).resolve().parent

# Load the saved model and class names
MODEL_PATH = ROOT_DIR / "potato_disease_model.keras"
METADATA_PATH = ROOT_DIR / "potato_disease_model.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

class_names = metadata["class_names"]
image_size = metadata["image_size"]

# Give the path of the image you want to test
image_path = input("Enter the full path of your leaf image: ").strip().strip('"')

# Load and preprocess the image
image = Image.open(image_path)
image = ImageOps.exif_transpose(image).convert("RGB")
image = image.resize((image_size, image_size))

image_array = np.array(image, dtype=np.float32)
image_array = np.expand_dims(image_array, axis=0)

# Predict
predictions = model.predict(image_array, verbose=0)[0]
predicted_index = int(np.argmax(predictions))

print("\n--- Prediction Results ---")
print("Predicted class:", class_names[predicted_index])
print(f"Confidence: {predictions[predicted_index] * 100:.2f}%")

print("\nAll class probabilities:")
for name, probability in zip(class_names, predictions):
    print(f"{name}: {probability * 100:.2f}%")
