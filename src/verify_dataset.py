import os

dataset_path = "dataset/phishing_urls.csv"  # Adjust if the path is different

if os.path.exists(dataset_path):
    print(f"✅ Dataset found: {dataset_path}")
else:
    print(f"❌ Dataset not found in {dataset_path}. Please check the file location.")
