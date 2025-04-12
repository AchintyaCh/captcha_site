import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# Configuration
MODEL_PATH = "captcha_model.h5"
CHARACTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'  # Update to your set
CAPTCHA_LENGTH = 5  # Update to your CAPTCHA's length
IMAGE_WIDTH, IMAGE_HEIGHT = 200, 50  # Resize target matching your model

# Load the model
model = load_model(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

def preprocess_image(image_bytes):
    """Convert uploaded image to model input format."""
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

def decode_prediction(pred):
    """Convert model output into text."""
    if isinstance(pred, list):
        return ''.join([CHARACTERS[np.argmax(p)] for p in pred])
    else:
        split_preds = np.split(pred[0], CAPTCHA_LENGTH)
        return ''.join([CHARACTERS[np.argmax(c)] for c in split_preds])

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    
    try:
        img = preprocess_image(image_bytes)
        prediction = model.predict(img)
        text = decode_prediction(prediction)
        return jsonify({"prediction": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "CAPTCHA Solver API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
