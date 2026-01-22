from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
import urllib.request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
MODEL_FILENAME = os.environ.get("MODEL_PATH", "deepfake_mobilenet.keras")
MODEL_URL = os.environ.get("MODEL_URL")  # optional: URL to download model if missing
BASE_DIR = os.path.dirname(__file__)
MODEL_FULL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# Lazy-loaded model
_model = None


def ensure_model_on_disk():
    """Ensure model file exists locally. If missing and MODEL_URL is set, try to download it."""
    if os.path.exists(MODEL_FULL_PATH):
        logging.info(f"Model file found at {MODEL_FULL_PATH}")
        return True

    if MODEL_URL:
        logging.info(f"Model file not found at {MODEL_FULL_PATH}. Attempting download from MODEL_URL.")
        try:
            os.makedirs(os.path.dirname(MODEL_FULL_PATH), exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_FULL_PATH)
            logging.info("Model downloaded successfully.")
            return True
        except Exception as e:
            logging.exception("Failed to download model from MODEL_URL")
            return False

    logging.warning(f"Model file not found at {MODEL_FULL_PATH} and no MODEL_URL provided.")
    return False


def get_model():
    """Load and return the Keras model, loading it lazily."""
    global _model
    if _model is not None:
        return _model

    if not ensure_model_on_disk():
        raise FileNotFoundError(f"Model not available at {MODEL_FULL_PATH}")

    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        logging.info(f"Loading model from {MODEL_FULL_PATH} ...")
        _model = load_model(MODEL_FULL_PATH)
        logging.info("Model loaded successfully.")
        return _model
    except Exception:
        logging.exception("Failed to load Keras model")
        raise


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", result="No image uploaded.")

    img_file = request.files["image"]
    if img_file.filename == "":
        return render_template("index.html", result="No image selected.")

    try:
        filename = secure_filename(img_file.filename)
        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        img_path = os.path.join(upload_folder, filename)
        img_file.save(img_path)

        from tensorflow.keras.preprocessing import image
        import numpy as np

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model = get_model()
        prediction = model.predict(img_array)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        if prediction < 0.5:
            label = "Likely AI Generated (Deepfake)"
            result_class = "fake"
        else:
            label = "Likely Real Image"
            result_class = "real"

        return render_template(
            "index.html",
            result=label,
            result_class=result_class,
            confidence=f"{confidence * 100:.2f}%",
            image_path='/' + img_path,
        )

    except Exception as e:
        logging.exception("Error during prediction")
        return render_template("index.html", result=f"Error: {str(e)}")


@app.route("/health")
def health():
    return jsonify({
        "model_path": MODEL_FULL_PATH,
        "model_exists": os.path.exists(MODEL_FULL_PATH),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)