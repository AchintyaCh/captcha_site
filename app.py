from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
         # Simulating prediction for demonstration
        file = request.files['file']
        filename = file.filename
        name_only = os.path.splitext(filename)[0] 
        return jsonify({
            'prediction': name_only
        })
    
    except Exception as e:
        # Log error to console for debugging
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
