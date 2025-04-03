import os
import sys
import joblib
import numpy as np
import tensorflow as tf

# Get the absolute path of the current directory (backend/src)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the OTHER 'src' directory (where feature_extraction.py is located)
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "src"))

# Add it to sys.path so Python can find 'feature_extraction.py'
sys.path.append(parent_dir)

print(f"Added {parent_dir} to sys.path")

# Import feature extraction module
try:
    from feature_extraction import extract_features
    print("Feature extraction module imported successfully!")
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(f"Error: Could not import 'extract_features'. Check if 'feature_extraction.py' is inside {parent_dir}.") from e

# Define paths to the scaler and model files
scaler_path = os.path.abspath(os.path.join(current_dir, "..", "..", "dataset", "scaler.pkl"))
model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "output", "NIDHI-221IT047-model.h5"))

print(f"🔍 Looking for scaler at: {scaler_path}")
print(f"🔍 Looking for model at: {model_path}")

# Check if the scaler and model files exist before loading
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f" Error: 'scaler.pkl' NOT found at {scaler_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f" Error: 'NIDHI-221IT047-model.h5' NOT found at {model_path}")

# Load the scaler and model
scaler = joblib.load(scaler_path)
model = tf.keras.models.load_model(model_path)

print(" Scaler and Model loaded successfully!")

# Function to predict phishing URL
# Function to predict phishing URL
def predict_url(url):
    print(f" Processing URL: {url}")  # Log received URL

    # Ensure the feature extraction is working
    features = extract_features(url)  
    if features is None:
        print(" Feature extraction failed!")
        return None  

    print(f" Extracted Features: {features}")  

    # Ensure the model & scaler are loaded correctly
    try:
        features = scaler.transform([features])
        prediction = model.predict(features)
        # Interpret prediction  
        predicted_class = "Phishing" if prediction[0][0] > 0.5 else "Safe"  # Adjust threshold as needed  
        print(f"Model Output: {prediction}, Predicted Class: {predicted_class}")  

        return predicted_class  
    except Exception as e:  
        print(f"Prediction Error: {e}")  
        return None  


# Accept URL input from the user
if __name__ == "__main__":
    url = input("\n🌐 Enter a URL to predict: ").strip()
    predict_url(url)
