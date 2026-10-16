import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "potato!" / "potato!"
MODEL_PATH = ROOT_DIR / "potato_disease_model.h5"
METADATA_PATH = ROOT_DIR / "potato_disease_model.json"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 123
EPOCHS = 12
FINE_TUNE_EPOCHS = 8
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dataset():
    random_generator = random.Random(SEED)
    class_directories = sorted(path for path in DATA_DIR.iterdir() if path.is_dir())
    if len(class_directories) < 2:
        raise ValueError(f"Expected at least two class folders in {DATA_DIR}")

    splits = {"train": [], "validation": [], "test": []}
    class_counts = {}
    for class_index, class_directory in enumerate(class_directories):
        image_paths = sorted(
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if len(image_paths) < 3:
            raise ValueError(
                f"Class {class_directory.name!r} needs at least 3 images; "
                f"found {len(image_paths)}"
            )

        random_generator.shuffle(image_paths)
        validation_count = max(1, round(len(image_paths) * 0.15))
        test_count = max(1, round(len(image_paths) * 0.15))
        splits["validation"].extend((str(path), class_index) for path in image_paths[:validation_count])
        splits["test"].extend(
            (str(path), class_index)
            for path in image_paths[validation_count : validation_count + test_count]
        )
        splits["train"].extend(
            (str(path), class_index)
            for path in image_paths[validation_count + test_count :]
        )
        class_counts[class_index] = len(image_paths) - validation_count - test_count

    return class_directories, splits, class_counts


def make_dataset(samples, training=False):
    paths, labels = zip(*samples)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if training:
        dataset = dataset.shuffle(
            len(samples), seed=SEED, reshuffle_each_iteration=True
        )

    def load_image(path, label):
        image_data = tf.io.read_file(path)
        image = tf.io.decode_image(image_data, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, label

    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_model(class_count):
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenet_preprocessing")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, backbone


def make_callbacks(initial_best=None):
    checkpoint_options = {}
    if initial_best is not None:
        checkpoint_options["initial_value_threshold"] = initial_best
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
            **checkpoint_options,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6
        ),
    ]


def main():
    tf.keras.utils.set_random_seed(SEED)
    class_directories, samples, class_counts = split_dataset()
    train_ds = make_dataset(samples["train"], training=True)
    validation_ds = make_dataset(samples["validation"])
    test_ds = make_dataset(samples["test"])
    class_names = [
        directory.name.split("___", maxsplit=1)[-1].replace("_", " ").title()
        for directory in class_directories
    ]
    total_train_images = sum(class_counts.values())
    class_weights = {
        index: total_train_images / (len(class_counts) * count)
        for index, count in class_counts.items()
    }

    print(f"Training images: {len(samples['train'])}")
    print(f"Validation images: {len(samples['validation'])}")
    print(f"Test images: {len(samples['test'])}")
    print(f"Classes: {class_names}")
    print(f"Training class weights: {class_weights}")

    model, backbone = build_model(len(class_names))
    print("\nTraining the classification head...")
    head_history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(),
    )
    best_validation_loss = min(head_history.history["val_loss"])

    print("\nFine-tuning the final MobileNetV2 layers...")
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(initial_best=best_validation_loss),
    )

    best_model = tf.keras.models.load_model(MODEL_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=0)
    true_labels = np.concatenate([labels.numpy() for _, labels in test_ds])
    predicted_labels = best_model.predict(test_ds, verbose=0).argmax(axis=1)
    recalls = {}
    for class_index, class_name in enumerate(class_names):
        class_mask = true_labels == class_index
        recalls[class_name] = float(
            np.mean(predicted_labels[class_mask] == class_index)
        )

    metadata = {
        "class_names": class_names,
        "image_size": list(IMAGE_SIZE),
        "confidence_threshold": 0.7,
        "test_images": len(samples["test"]),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "recall_by_class": recalls,
        "seed": SEED,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nHeld-out test accuracy: {test_accuracy:.3f}")
    print(f"Held-out recall by class: {json.dumps(recalls, indent=2)}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved model metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
