import pandas as pd
import re
from urllib.parse import urlparse

# Load the dataset
input_file = "output/virus_total_results.csv"
output_file = "output/221IT047_URLfeaturedataset.csv"

df = pd.read_csv(input_file)

# Define phishing-related words
phishing_keywords = ['login', 'admin', 'secure', 'bank', 'account', 'verify', 'webscr', 'password']

def extract_features(url):
    features = {}
    parsed_url = urlparse(url)
    
    # Structural Features
    features["url_length"] = len(url)
    features["hostname_length"] = len(parsed_url.netloc)
    features["path_length"] = len(parsed_url.path)
    features["num_dots"] = url.count('.')
    features["num_hyphens"] = url.count('-')
    features["num_slashes"] = url.count('/')
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_special_chars"] = sum(c in ['@', '?', '&', '=', '_'] for c in url)

    # Lexical Features
    features["has_ip"] = int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', parsed_url.netloc)))
    features["has_https"] = int(parsed_url.scheme == 'https')
    features["num_subdomains"] = parsed_url.netloc.count('.') - 1
    features["contains_phishing_words"] = int(any(word in url.lower() for word in phishing_keywords))

    return features

# Apply feature extraction to all URLs
feature_df = df["URL"].apply(lambda x: extract_features(str(x))).apply(pd.Series)

# Add labels
feature_df["label"] = df["label"]

# Save the extracted features
feature_df.to_csv(output_file, index=False)
print(f"✅ Feature extraction complete! Dataset saved at {output_file}")
