import os
from src.feature_extraction import extract_features
import pandas as pd

# Load dataset
dataset_path = "output/virus_total_results.csv"
df = pd.read_csv(dataset_path)

# Extract features
features_df = df["URL"].apply(lambda x: pd.Series(extract_features(x)))

# Save the processed dataset
output_path = "dataset/extracted_features.csv"
features_df.to_csv(output_path, index=False)

print(f"Feature extraction complete. File saved at {output_path}")
