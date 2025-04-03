import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, confusion_matrix, roc_curve, auc
)

# Create output directory
os.makedirs("output", exist_ok=True)

# Start measuring training time
train_start_time = time.time()

# Load the training data
X_train = pd.read_csv("dataset/X_train.csv").values
y_train = pd.read_csv("dataset/y_train.csv").values

# Load the testing data (Unseen dataset)
X_test = pd.read_csv("dataset/X_test.csv").values
y_test = pd.read_csv("dataset/y_test.csv").values

# Convert y_train and y_test to categorical if needed
if len(np.unique(y_train)) > 2:
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

# Define the ANN Model with Batch Normalization
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  
    tf.keras.layers.Dense(128, activation='relu', name="dense_1"),
    tf.keras.layers.BatchNormalization(name="batch_norm_1"),
    tf.keras.layers.Dense(64, activation='relu', name="dense_2"),
    tf.keras.layers.BatchNormalization(name="batch_norm_2"),
    tf.keras.layers.Dense(32, activation='relu', name="dense_3"),
    tf.keras.layers.BatchNormalization(name="batch_norm_3"),
    tf.keras.layers.Dense(1, activation='sigmoid', name="output")  
])

# Print model summary (Displays parameters layer-by-layer)
model.summary()

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# End training time measurement
train_end_time = time.time()
training_time = train_end_time - train_start_time
print(f"Training Time: {training_time:.4f} seconds")

# Save training time to Excel
train_time_df = pd.DataFrame({"Training Time (s)": [training_time]})
train_time_df.to_excel("output/SNEHA-221IT063-trainingtime.xlsx", index=False)

# Save trained model
model.save("output/NIDHI-221IT047-model.h5")
print("✅ Model saved successfully.")

# Save Training Accuracy & Loss for plotting
training_history_df = pd.DataFrame({
    "Epoch": list(range(1, 21)),
    "Accuracy": history.history["accuracy"],
    "Loss": history.history["loss"]
})
training_history_df.to_excel("output/SNEHA-221IT063-training-history.xlsx", index=False)

# Plot & Save Accuracy Graph
plt.figure(figsize=(8, 6))
plt.plot(training_history_df["Epoch"], training_history_df["Accuracy"], marker='o', label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epochs")
plt.legend()
plt.grid(True)
plt.savefig("output/SNEHA-221IT063-accuracygraph.jpeg")
plt.show()

# Plot & Save Loss Graph
plt.figure(figsize=(8, 6))
plt.plot(training_history_df["Epoch"], training_history_df["Loss"], marker='o', color='red', label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs. Epochs")
plt.legend()
plt.grid(True)
plt.savefig("output/SNEHA-221IT063-lossgraph.jpeg")
plt.show()

# Load the trained model
loaded_model = tf.keras.models.load_model("output/NIDHI-221IT047-phishing-model.h5")

# Start measuring testing time
test_start_time = time.time()

# Perform batch prediction
y_pred_prob = loaded_model.predict(X_test, batch_size=32)
y_pred_labels = (y_pred_prob > 0.5).astype(int).flatten()

# End testing time measurement
test_end_time = time.time()
total_testing_time = test_end_time - test_start_time
average_testing_time = total_testing_time / len(X_test)

print(f"Total Samples: {len(X_test)}")
print(f"Total Testing Time: {total_testing_time:.4f} seconds")
print(f"Average Testing Time per Sample: {average_testing_time:.6f} seconds")

# Save testing time results to an Excel file
test_time_df = pd.DataFrame({
    "Total Time (s)": [total_testing_time],
    "Avg Time per Sample (s)": [average_testing_time]
})
test_time_df.to_excel("output/SNEHA-221IT063-testingtime.xlsx", index=False)

# Save Predictions to Excel
predictions_df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred_labels})
predictions_df.to_excel("output/SNEHA-221IT063-prediction.xlsx", index=False)

# Performance Metrics
accuracy = accuracy_score(y_test, y_pred_labels)
precision = precision_score(y_test, y_pred_labels)
recall = recall_score(y_test, y_pred_labels)
f1 = f1_score(y_test, y_pred_labels)
mcc = matthews_corrcoef(y_test, y_pred_labels)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")

# Save Metrics to Excel
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "MCC"],
    "Value": [accuracy, precision, recall, f1, mcc]
})
metrics_df.to_excel("output/NIDHI-221IT047-metrics.xlsx", index=False)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:\n", conf_matrix)

# Save confusion matrix to Excel
conf_matrix_df = pd.DataFrame(conf_matrix, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"])
conf_matrix_df.to_excel("output/NIDHI-221IT047-confusion-matrix.xlsx")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("output/NIDHI-221IT047-rocgraph.jpeg")
plt.show()

print("✅ All outputs saved successfully!")
