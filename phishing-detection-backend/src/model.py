import os
import joblib
import tensorflow as tf

# Define paths for model and scaler
MODEL_PATH = os.path.abspath("output\NIDHI-221IT047-model.h5")
SCALER_PATH = os.path.abspath("dataset\scaler.pkl")

# Load the scaler and model
print(" Loading model and scaler...")
scaler = joblib.load(SCALER_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model and scaler loaded successfully!")

# Function to return model and scaler
def get_model_and_scaler():
    return model, scaler
