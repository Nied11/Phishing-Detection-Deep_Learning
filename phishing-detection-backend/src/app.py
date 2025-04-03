
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model  # Import TensorFlow model loader
import joblib  # Still needed for scaler

# Fix path issues for feature_extraction module
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(BASE_DIR, "src"))

import feature_extraction  # Import after fixing path

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
try:
    model = load_model("output/NIDHI-221IT047-model.h5")  # Load .h5 model
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None

# Load the scaler
try:
    scaler = joblib.load("dataset/scaler.pkl")
    print(" Scaler loaded successfully!")
except Exception as e:
    print(f" Error loading scaler: {e}")
    scaler = None

@app.route('/')
def home():
    return " Phishing URL Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded properly"}), 500

    try:
        data = request.json
        url = data.get("url", "").strip()

        if not url:
            return jsonify({"error": "No URL provided"}), 400

        # Extract features from the URL
        features = feature_extraction.extract_features(url)  # Your custom function
        features = np.array(features).reshape(1, -1)  # Reshape for model input
        
        # Normalize features
        features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features)[0][0]  # Extract single value
        prediction_label = "Phishing" if prediction >= 0.5 else "Legitimate"  # Adjust based on threshold

        return jsonify({"url": url, "prediction": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
