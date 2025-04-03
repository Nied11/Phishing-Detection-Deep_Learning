
import pandas as pd

# Load the dataset (replace with your actual dataset file)
df = pd.read_csv("dataset\phishing_urls.csv")

# Convert numeric columns only
df_numeric = df.select_dtypes(include=['number'])

# Fill missing values in numeric columns with their mean
df_numeric.fillna(df_numeric.mean(), inplace=True)

# Handle non-numeric columns separately
df_non_numeric = df.select_dtypes(exclude=['number'])
df_non_numeric.fillna(df_non_numeric.mode().iloc[0], inplace=True)  # Fill categorical with mode

# Combine both numeric and non-numeric parts back
df_cleaned = pd.concat([df_numeric, df_non_numeric], axis=1)

# Save the preprocessed dataset
df_cleaned.to_csv("output/221IT063-Pre-processed_Dataset.csv", index=False)

print("Preprocessed dataset saved successfully!")
