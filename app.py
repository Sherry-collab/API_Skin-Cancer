import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your trained skin cancer detection model
model = load_model('skin_cancer_model.h5')

# Class labels (update if needed)
CLASS_NAMES = [
    'Actinic Keratoses',
    'Basal Cell Carcinoma',
    'Benign Keratosis-like Lesions',
    'Dermatofibroma',
    'Melanocytic Nevi',
    'Melanoma',
    'Vascular Lesions'
]

@app.route('/')
def home():
    return "API is running", 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Load and preprocess the image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((28, 28))  # Match input size of your model
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 28, 28, 3)

        # Get prediction
        predictions = model.predict(img_array)

        if len(predictions) == 0 or len(predictions[0]) != len(CLASS_NAMES):
            return jsonify({
                'error': 'Mismatch between model output and CLASS_NAMES',
                'prediction_shape': str(predictions[0].shape),
                'class_names_count': len(CLASS_NAMES)
            }), 500

        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions[0]))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
