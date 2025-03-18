from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from feature_extraction import extract_features

app = Flask(__name__)
model = tf.keras.models.load_model("models/phishing_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url_features = extract_features(data['url'])
    prediction = model.predict(np.array([[url_features['url_length'], url_features['contains_https'], url_features['special_char_count']]]))
    return jsonify({'phishing_probability': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
