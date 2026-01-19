from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "deepfake_mobilenet.keras"
model = load_model(MODEL_PATH)

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# About page route
@app.route("/about")
def about():
    return render_template("about.html")

# Prediction route
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

        # Preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        if prediction < 0.5:
            label = "AI Generated (Deepfake)"
            result_class = "fake"
        else:
            label = "Real Image"
            result_class = "real"

        return render_template("index.html",
                               result=label,
                               result_class=result_class,
                               confidence=f"{confidence * 100:.2f}%",
                               image_path='/' + img_path)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
