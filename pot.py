import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ==========================
# 1. Check Directory
# ==========================

print("Current Working Directory:", os.getcwd())
print("Files in Directory:", os.listdir())

# ==========================
# 2. Load Dataset
# ==========================

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "potato!/potato!",   # IMPORTANT: correct dataset path
    shuffle=True,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
print("Classes:", class_names)
print("Total Batches:", len(dataset))


# ==========================
# 3. Split Dataset
# ==========================

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1):

    ds_size = len(ds)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

print("Train batches:", len(train_ds))
print("Validation batches:", len(val_ds))
print("Test batches:", len(test_ds))


# ==========================
# 4. Optimize Dataset
# ==========================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ==========================
# 5. Rescaling
# ==========================

resize_and_rescale = tf.keras.Sequential([
    layers.Rescaling(1./255)
])


# ==========================
# 6. Data Augmentation
# ==========================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])


# ==========================
# 7. Visualize Images
# ==========================

plt.figure(figsize=(10,10))

for images, labels in train_ds.take(1):
    for i in range(9):

        ax = plt.subplot(3,3,i+1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[labels[i]])

        plt.axis("off")

plt.show()


# ==========================
# 8. Build CNN Model
# ==========================

IMAGE_SIZE = 256
CHANNELS = 3
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

n_classes = len(class_names)

model = tf.keras.Sequential([

    resize_and_rescale,
    data_augmentation,

    layers.Conv2D(32,(3,3),activation="relu",input_shape=input_shape),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(64,activation="relu"),

    layers.Dense(n_classes,activation="softmax")
])


# ==========================
# 9. Compile Model
# ==========================

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)


# ==========================
# 10. Train Model
# ==========================

EPOCHS = 20

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)


# ==========================
# 11. Evaluate Model
# ==========================

scores = model.evaluate(test_ds)

print("Test Loss:", scores[0])
print("Test Accuracy:", scores[1])


# ==========================
# 12. Save Model
# ==========================

model.save("potato_disease_model.h5")

print("Model Saved Successfully")


