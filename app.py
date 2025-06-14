from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Model loading with error handling
try:
    # Debugging: Print directory structure
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Files present: {os.listdir('.')}")
    
    model_path = os.path.join(os.path.dirname(__file__), "skin_cancer.keras")
    logger.info(f"Loading model from: {model_path}")
    
    model = load_model(model_path, compile=False)
    logger.info("✅ Model loaded successfully!")
    
    # Verify model input shape (important for preprocessing)
    logger.info(f"Model input shape: {model.input_shape}")
    
except Exception as e:
    logger.error(f"❌ Model loading failed: {str(e)}")
    raise  # Crash early if model can't load

# ✅ Enhanced image preprocessing
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Check if image needs conversion (e.g., RGBA to RGB)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Use model's expected input shape instead of hardcoding
        target_size = model.input_shape[1:3]  # Get (height, width) from model
        img = img.resize(target_size)
        
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, *target_size, 3)  # Use model's shape
        
        return img_array
        
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise

# ✅ Predict route with enhanced validation
@app.route("/predict", methods=["POST"])
def predict():
    # Check for file existence
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    # Validate file
    if file.filename == '':
        logger.warning("Empty file submitted")
        return jsonify({"error": "No selected file"}), 400
        
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({"error": "Only image files are allowed"}), 400

    try:
        # Read and verify file size
        file_bytes = file.read()
        if len(file_bytes) == 0:
            logger.warning("Empty file content")
            return jsonify({"error": "Empty file content"}), 400
            
        # Process and predict
        img_array = preprocess_image(file_bytes)
        prediction = model.predict(img_array)
        
        # Get results
        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        logger.info(f"Prediction successful - Class: {class_index}, Confidence: {confidence:.2f}")
        
        return jsonify({
            "predicted_class": class_index,
            "confidence": confidence,
            "model_input_shape": model.input_shape[1:]  # For debugging
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# ✅ Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)