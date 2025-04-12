import os
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("captcha_model.h5")

# CAPTCHA configuration
CHARACTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
CAPTCHA_LENGTH = 5
IMAGE_SIZE = (128, 128)

app = Flask(__name__)

def preprocess_image_bytes(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def decode_prediction(prediction):
    if isinstance(prediction, list):
        return ''.join([CHARACTERS[np.argmax(p)] for p in prediction])
    else:
        return ''.join([CHARACTERS[np.argmax(c)] for c in np.split(prediction[0], CAPTCHA_LENGTH)])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()
        img = preprocess_image_bytes(image_bytes)
        prediction = model.predict(img)
        predicted_text = decode_prediction(prediction)
        return jsonify({"prediction": predicted_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
