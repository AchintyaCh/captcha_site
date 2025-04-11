from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# Define the same model architecture used during training
class CaptchaModel(nn.Module):
    def __init__(self, num_classes, max_length):
        super(CaptchaModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * 32 * 32, max_length * num_classes)
        self.num_classes = num_classes
        self.max_length = max_length

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(-1, self.max_length, self.num_classes)
        return x

# Configuration
NUM_CLASSES = 36  # 0-9 + A-Z (adjust if needed)
MAX_LENGTH = 5    # Number of characters in each CAPTCHA

# Create model and load state_dict
model = CaptchaModel(NUM_CLASSES, MAX_LENGTH)
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
model.eval()

# Character decoding
characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
index_to_char = {i: char for i, char in enumerate(characters)}

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        predictions = torch.argmax(output, dim=2)
        predicted_label = ''.join(index_to_char[int(p)] for p in predictions[0])

    return jsonify({'prediction': predicted_label})

@app.route('/', methods=['GET'])
def home():
    return 'CAPTCHA model API is running!'

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
