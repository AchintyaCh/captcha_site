from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = torch.load('model.pt', map_location='cpu')
model.eval()

# Preprocessing (adjust based on your model)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Character map used in training
characters = '0123456789abcdefghijklmnopqrstuvwxyz'  # update based on your model
max_length = 5  # adjust this based on label length

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file.stream)
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
    
    prediction = ''
    for output in outputs:
        pred_idx = torch.argmax(output, dim=1)
        prediction += characters[pred_idx]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
