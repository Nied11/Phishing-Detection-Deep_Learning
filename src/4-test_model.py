import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Load the trained model
model_path = "output/NIDHI-221IT047-model.h5"
model = tf.keras.models.load_model(model_path)
print(f"Loaded model from '{model_path}'.")

# Load test data
X_test = pd.read_csv("dataset/X_test.csv").values
y_test = pd.read_csv("dataset/y_test.csv").values.ravel()  # Convert to 1D array

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()  # Convert to 1D array

# Count misclassified instances
misclassified_count = np.sum(y_pred != y_test)

# Print actual vs predicted labels (limited for readability)
print("\n🔹 Actual vs Predicted Labels (first 10 samples):")
for i in range(min(10, len(y_test))):
    print(f"Actual: {y_test[i]} | Predicted: {y_pred[i]}")

print(f"\n Total Misclassified: {misclassified_count} / {len(y_test)}")

# Save predictions to Excel file
os.makedirs("output", exist_ok=True)
df = pd.DataFrame({"Actual Label": y_test, "Predicted Label": y_pred})
prediction_file = "output/NIDHI-221IT047-prediction.xlsx"
df.to_excel(prediction_file, index=False)
print(f"Predictions saved in '{prediction_file}'.")
