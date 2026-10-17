
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# 1. SETTINGS
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "potato!" / "potato!"

IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
SEED = 12

MODEL_PATH = ROOT_DIR / "potato_disease_model.keras"
METADATA_PATH = ROOT_DIR / "potato_disease_model.json"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================
# 2. LOAD AND SPLIT DATASET
# ==========================================

def split_dataset():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    # Folder names are sorted alphabetically, as in
    # image_dataset_from_directory().
    folder_names = sorted(
        folder.name
        for folder in DATA_DIR.iterdir()
        if folder.is_dir()
    )

    if len(folder_names) != 3:
        raise ValueError(
            f"Expected 3 class folders, found: {folder_names}"
        )

    # Keep the class order consistent with your sir's notebook.
    expected_folders = [
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
    ]

    if folder_names != expected_folders:
        raise ValueError(
            "Unexpected class folders.\n"
            f"Found: {folder_names}\n"
            f"Expected: {expected_folders}"
        )

    # These names correspond to the same indices used by the CNN.
    class_names = [
        "Potato_Early_blight",
        "Potato_Late_blight",
        "Potato_healthy",
    ]

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    train_samples = []
    val_samples = []
    test_samples = []

    # Split EACH class separately to avoid class imbalance.
    for label, folder_name in enumerate(folder_names):
        folder = DATA_DIR / folder_name

        files = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

        if len(files) < 10:
            raise ValueError(
                f"Too few images in {folder_name}: {len(files)}"
            )

        random.Random(SEED + label).shuffle(files)

        total = len(files)
        train_end = int(total * 0.80)
        val_end = int(total * 0.90)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        train_samples.extend(
            (str(path), label) for path in train_files
        )
        val_samples.extend(
            (str(path), label) for path in val_files
        )
        test_samples.extend(
            (str(path), label) for path in test_files
        )

        print(
            f"{class_names[label]}: "
            f"{len(train_files)} train, "
            f"{len(val_files)} validation, "
            f"{len(test_files)} test"
        )

    # Shuffle training samples only.
    random.Random(SEED).shuffle(train_samples)

    print("\nTotal images:")
    print("Training:", len(train_samples))
    print("Validation:", len(val_samples))
    print("Testing:", len(test_samples))

    return class_names, folder_names, train_samples, val_samples, test_samples


# ==========================================
# 3. PREPROCESS IMAGES
# ==========================================

def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    return image, label


def make_dataset(samples, training=False):
    paths = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(samples),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ==========================================
# 4. BUILD YOUR SIR'S CNN
# ==========================================

def build_model(number_of_classes):
    # Resizing and rescaling are inside the model.
    # The API must supply RGB pixels in the 0-255 range.
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.Rescaling(1.0 / 255),
    ], name="resize_and_rescale")

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ], name="data_augmentation")

    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

        resize_and_rescale,
        data_augmentation,

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(number_of_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        ),
        metrics=["accuracy"],
    )

    return model


# ==========================================
# 5. TRAIN AND EVALUATE
# ==========================================

def main():
    print("TensorFlow version:", tf.__version__)
    print("Dataset path:", DATA_DIR)

    class_names, folder_names, train_samples, val_samples, test_samples = (
        split_dataset()
    )

    train_ds = make_dataset(train_samples, training=True)
    val_ds = make_dataset(val_samples)
    test_ds = make_dataset(test_samples)

    model = build_model(len(class_names))
    model.summary()

    checkpoint = ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint],
        verbose=1,
    )

    # Load the best validation-accuracy checkpoint.
    best_model = tf.keras.models.load_model(MODEL_PATH)

    print("\nEvaluating on the separate test set...")
    test_loss, test_accuracy = best_model.evaluate(
        test_ds,
        verbose=1
    )

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # Get predictions in the same order as test samples.
    predictions = best_model.predict(test_ds, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.array(
        [label for _, label in test_samples]
    )

    print("\nClassification report:")
    print(classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
    ))

    # Save the labels and settings alongside the model.
    metadata = {
        "class_names": class_names,
        "folder_names": folder_names,
        "image_size": IMAGE_SIZE,
        "confidence_threshold": 0.70,
        "test_accuracy": float(test_accuracy),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nTraining completed.")
    print("Model saved to:", MODEL_PATH)
    print("Metadata saved to:", METADATA_PATH)


if __name__ == "__main__":
    main()
