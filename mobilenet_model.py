import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_PATH = "deepfake_mobilenet.h5"

model = tf.keras.models.load_model("deepfake_mobilenet.h5")


def predict_image(image_path):
    try:
        img = cv2.imread(image_path)

        if img is None:
            return "Invalid Image", 0.0

        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0][0]

        if pred > 0.5:
            return "FAKE", round(pred * 100, 2)
        else:
            return "REAL", round((1 - pred) * 100, 2)

    except Exception as e:
        print("Prediction error:", e)
        return "Error", 0.0
