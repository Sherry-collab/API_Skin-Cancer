from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Load model once at the start
model = load_model("skin cancer.h5")

# Preprocess incoming image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 28, 28, 3)

# API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files['file']
        img_array = preprocess_image(file.read())
        prediction = model.predict(img_array)
        class_index = int(np.argmax(prediction))
        confidence = round(float(np.max(prediction)) * 100, 2)

        return jsonify({
            "predicted_class": class_index,
            "confidence": f"{confidence} %"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
