import os
import uuid
import cv2
import torch
from flask import Flask, request, render_template, jsonify
from cnn_captcha_solver.segmenter import Segmenter
from cnn_captcha_solver.model import Model
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Load your model
model = Model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('Model_state_dict.pt', map_location=device))
model.to(device)
model.eval()

# Load segmenter
segmenter = Segmenter()

# Load label dictionary
label_key_path = os.path.join(os.path.dirname(__file__), 'label_key.csv')
with open(label_key_path, 'r') as f:
    label_dict = {int(line.split(',')[1]): line.split(',')[0] for line in f.read().splitlines()[1:]}

# Match CharsDataset transformations
transform = transforms.Compose([
    transforms.Grayscale(),                  # Convert to 1-channel grayscale
    transforms.Resize((15, 12)),             # Match training size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))     # Same normalization
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['file']
    filename = f"{uuid.uuid4().hex}.png"
    temp_path = os.path.join('/tmp', filename)

    try:
        image_file.save(temp_path)

        img = cv2.imread(temp_path)
        if img is None:
            raise ValueError("Failed to read image.")

        # Pass the temporary file path to segmenter
        segmented_chars = segmenter.segment_chars(temp_path, plot=False)
        predicted_chars = []

        for char_img, _ in segmented_chars:
            pil_img = Image.fromarray(char_img)
            char_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(char_tensor)
                predicted_label = torch.argmax(output).item()

            predicted_chars.append(label_dict[predicted_label])

        predicted_captcha = ''.join(predicted_chars)
        return jsonify({'prediction': predicted_captcha})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
