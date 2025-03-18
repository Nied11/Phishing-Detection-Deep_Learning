import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

# Load trained model
model = tf.keras.models.load_model("models/rnn_model.keras")

# Load test dataset
scaler = joblib.load("dataset/scaler.pkl")
df = pd.read_csv("output/221IT063-Pre-processed_Dataset.csv")
df.columns = df.columns.str.lower()

df["label"] = df["label"].map({"benign": 0, "phishing": 1})

y_test = df["label"].astype(int).values
X_test = df.drop(columns=["label"])
X_test = X_test.select_dtypes(include=[np.number])
X_test = scaler.transform(X_test)
X_test = np.expand_dims(X_test, axis=1)  # Reshape for LSTM

# Generate predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Compute Evaluation Metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
roc_score = roc_auc_score(y_test, y_pred)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_score:.4f}")

# Plot Confusion Matrix with labels
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Phishing"], yticklabels=["Benign", "Phishing"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

