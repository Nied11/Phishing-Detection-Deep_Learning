import pandas as pd

dataset_path = "dataset/phishing_urls.csv"  # Adjust if necessary

# Load the dataset
df = pd.read_csv(dataset_path)

# Show basic information
print("Dataset Preview:")
print(df.head())  # Show first 5 rows

print("\nDataset Info:")
print(df.info())  # Show column details

print("\nNull Values:")
print(df.isnull().sum())  # Check for missing values
