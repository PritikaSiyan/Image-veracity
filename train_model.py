import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from collections import Counter
import kagglehub

# -----------------------------
# Load Dataset
# -----------------------------
DATASET_DIR = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
DATASET_DIR = os.path.join(DATASET_DIR, "Dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
VAL_DIR = os.path.join(DATASET_DIR, "Validation")
TEST_DIR = os.path.join(DATASET_DIR, "Test")

# -----------------------------
# Parameters
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

# -----------------------------
# Data Augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# -----------------------------
# Data Generators
# -----------------------------
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)


val_data = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)
print(train_data.class_indices)

# -----------------------------
# Class Weights
# -----------------------------
counter = Counter(train_data.classes)
total = sum(counter.values())
class_weight = {
    0: total / (2 * counter[0]),
    1: total / (2 * counter[1])
}
print("Class weights:", class_weight)

# -----------------------------
# Load MobileNetV2 base
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# -----------------------------
# Build Model
# -----------------------------
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True)

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# -----------------------------
# Evaluate on Test Data
# -----------------------------
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")

# -----------------------------
# Save Final Model
# -----------------------------
model.save("deepfake_mobilenet_final.keras")
print("Model saved as deepfake_mobilenet_final.keras")

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()