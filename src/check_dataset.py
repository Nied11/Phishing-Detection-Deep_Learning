import pandas as pd

# Path to the dataset
file_path = "dataset/phishing_urls.csv"  # Adjust path if needed

# Load dataset
df = pd.read_csv(file_path)

# Display first few rows
print(df.head())

# Print column names
print("\nColumn names:", df.columns)
