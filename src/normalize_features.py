import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("output/virus_total_results.csv")

# Identify numerical columns (excluding categorical/text columns)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Apply Min-Max Normalization (0 to 1 scaling)
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save the normalized dataset
df.to_csv("dataset/extracted_features_normalized.csv", index=False)

print("Normalization applied to numerical columns:")
print(num_cols.tolist())
print("Normalized dataset saved as extracted_features_normalized.csv")
