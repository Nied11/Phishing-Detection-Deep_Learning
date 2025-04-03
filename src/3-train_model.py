import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Load Data with Proper Shape Checking
X_train = pd.read_csv("dataset/X_train.csv")
y_train = pd.read_csv("dataset/y_train.csv")

# Ensure No Extra Headers or Empty Rows
print(f"X_train shape before cleanup: {X_train.shape}")
print(f"y_train shape before cleanup: {y_train.shape}")

#  Fix Mismatched Lengths
min_samples = min(len(X_train), len(y_train))
X_train = X_train.iloc[:min_samples, :]  # Trim excess rows
y_train = y_train.iloc[:min_samples]  # Trim excess rows

#  Ensure `y_train` is a 1D array (for binary classification)
y_train = y_train.values.reshape(-1, 1)

print(f" After Fix: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

#  Classification Type Handling
num_classes = len(np.unique(y_train))
if num_classes > 2:
    y_train = tf.keras.utils.to_categorical(y_train)
    output_activation = "softmax"
    loss_function = "categorical_crossentropy"
    output_units = num_classes
else:
    output_activation = "sigmoid"
    loss_function = "binary_crossentropy"
    output_units = 1

#  Define ANN Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(output_units, activation=output_activation)
])

#  Compile Model
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

#  Display Model Summary
model.summary()

#  Training Callback for Logging
class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_filename="training_logs.csv"):
        self.log_filename = log_filename
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.logs.append([epoch + 1, logs.get("accuracy", 0), logs.get("loss", 0), logs.get("val_accuracy", 0), logs.get("val_loss", 0)])
        print(f"Epoch {epoch + 1}: Accuracy={logs.get('accuracy', 0):.4f}, Loss={logs.get('loss', 0):.4f}")

    def on_train_end(self, logs=None):
        df_logs = pd.DataFrame(self.logs, columns=["Epoch", "Train Accuracy", "Train Loss", "Val Accuracy", "Val Loss"])
        df_logs.to_csv(self.log_filename, index=False)
        print(f"📊 Training logs saved as '{self.log_filename}'.")

#  Train Model
training_callback = TrainingCallback()
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[training_callback])

#  Save Model
os.makedirs("output", exist_ok=True)
model_path = "output/NIDHI-221IT047-model.h5"
model.save(model_path)

print(f" Model training complete! Model saved as '{model_path}'.")
