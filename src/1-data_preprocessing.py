import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import os

# Load dataset
df = pd.read_csv("output/221IT063-Pre-processed_Dataset.csv")

# Convert all column names to lowercase (avoid case sensitivity issues)
df.columns = df.columns.str.lower()

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Handle missing values:
# - Numeric columns: Fill missing values with mean
# - Categorical columns: Fill missing values with mode (most frequent value)
for col in df.columns:
    if col in categorical_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(numeric_only=True), inplace=True)

# Encode categorical columns
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Separate features and labels
X = df.drop(columns=["label"])  # Replace "label" with your actual target column name
y = df["label"]

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset (before SMOTE to prevent data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Apply SMOTE for balancing classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 🔹 Ensure X_train and y_train have the same length
min_length = min(len(X_train), len(y_train))
X_train = X_train[:min_length]
y_train = y_train[:min_length]

# Check the label distribution after SMOTE
print("Label distribution after SMOTE:")
print(pd.Series(y_train).value_counts())

# Ensure dataset folder exists
os.makedirs("dataset", exist_ok=True)

# Save scaler
joblib.dump(scaler, "dataset/scaler.pkl")
print("Scaler saved as scaler.pkl")

# Save preprocessed data
pd.DataFrame(X_train).to_csv("dataset/X_train.csv", index=False)
pd.DataFrame(y_train).to_csv("dataset/y_train.csv", index=False)
pd.DataFrame(X_test).to_csv("dataset/X_test.csv", index=False)
pd.DataFrame(y_test).to_csv("dataset/y_test.csv", index=False)

print("✅ Data Preprocessing Completed & Saved!")
