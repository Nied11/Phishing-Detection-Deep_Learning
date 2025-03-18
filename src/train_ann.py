import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Load dataset
df = pd.read_csv("output/221IT063-Pre-processed_Dataset.csv")  # Ensure correct file path

# --------------------------------------
# ✅ DATA CLEANING AND PREPROCESSING
# --------------------------------------

# Drop non-numeric columns
non_numeric_cols = ["FILENAME", "URL", "Domain", "TLD", "Title"]
existing_cols = [col for col in non_numeric_cols if col in df.columns]
df = df.drop(columns=existing_cols, errors='ignore')  # Ignore errors if columns don't exist

# Check missing values in labels
print("Missing label values before cleanup:", df["label"].isnull().sum())

# Drop rows where 'label' is missing
df = df.dropna(subset=["label"])

# Convert labels to numeric
df["label"] = df["label"].map({"benign": 0, "phishing": 1})

# Verify dataset size
print(f"Final dataset size after dropping missing labels: {df.shape}")
print("Label distribution:\n", df["label"].value_counts())

# Convert all remaining values to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Drop remaining NaN rows
df = df.dropna()

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Final check
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

if X.empty or y.empty:
    print("❌ Error: No valid data for training!")
    exit()

# --------------------------------------
# ✅ TRAIN-TEST SPLIT & NORMALIZATION
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features (optional, but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------
# ✅ ANN MODEL TRAINING (EXAMPLE)
# --------------------------------------
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),  
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train with early stopping
model.fit(X_train, y_train, epochs=100,batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

