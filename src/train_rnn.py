import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load preprocessed dataset
df = pd.read_csv("output/221IT063-Pre-processed_Dataset.csv")
df.columns = df.columns.str.lower()

# Debug: Print column names to check for 'label'
print("Dataset Columns:", df.columns)

# Ensure 'label' exists before processing
if "label" in df.columns:
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["label"])  # Remove rows with NaN labels
    df = df[df["label"].isin(["benign", "phishing"])]  # Ensure only expected labels are present
    df["label"] = df["label"].map({"benign": 0, "phishing": 1})
    
    if df["label"].isnull().any():
        print("Unmapped labels:", df[df["label"].isnull()]
              ["label"].unique())
        raise ValueError("Some labels could not be mapped. Check for unexpected values in the 'label' column.")
    
    y = df["label"].astype(int).values
    X = df.drop(columns=["label"])
else:
    raise KeyError("The column 'label' was not found in the dataset. Check the CSV file.")

# Print column data types before filtering
print("Feature Data Types Before Filtering:\n", X.dtypes)

# Remove non-numeric columns from X
X = X.select_dtypes(include=[np.number])

# Ensure there are features left after filtering
if X.shape[1] == 0:
    raise ValueError("No numeric features found in the dataset. Check your preprocessing step.")

# Debug: Print the remaining features after filtering
print("Features selected for training:", X.columns.tolist())

# Convert to numpy array
X = X.values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for inference
joblib.dump(scaler, "dataset/scaler.pkl")

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Reshape for RNN (LSTM requires 3D input: samples, time steps, features)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# Debug: Print final dataset shape before training
print(f"Final Training Data Shape: {X_train.shape}, Labels Shape: {y_train.shape}")
print(f"Final Testing Data Shape: {X_test.shape}, Labels Shape: {y_test.shape}")

# RNN Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    
    LSTM(128, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save model
model.save("models/rnn_model.keras")
print("Model saved successfully!")
