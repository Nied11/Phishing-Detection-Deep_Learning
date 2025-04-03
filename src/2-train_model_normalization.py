import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ✅ Step 1: Load the feature dataset
data_file = "output/221IT047_URLfeaturedataset.csv"
data = pd.read_csv(data_file)

# 🔍 Ensure 'label' column exists
if "label" not in data.columns:
    raise ValueError("🚨 ERROR: 'label' column is missing in the dataset!")

# ✅ Step 2: Separate features (X) and labels (y)
X = data.drop(columns=["label"])  # Features
y = data["label"]  # Labels

# 🔍 Ensure no missing values
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    raise ValueError("🚨 ERROR: Missing values detected in dataset!")

# ✅ Step 3: Validate dataset shape
if X.shape[0] != y.shape[0]:
    raise ValueError(f"🚨 ERROR: X has {X.shape[0]} samples, but y has {y.shape[0]}!")

print(f"✅ Data Loaded Successfully! X shape: {X.shape}, y shape: {y.shape}")

# ✅ Step 4: Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Step 6: Build the Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# ✅ Step 7: Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Step 8: Train the Model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ✅ Step 9: Save Model & Scaler
model.save("output/NIDHI-221IT047-model.h5")
np.save("output/scaler.npy", scaler.mean_)  # Save scaler parameters

print("🎉 Model training complete! ✅ Model and scaler saved successfully.")
