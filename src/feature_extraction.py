import re
import tldextract
import numpy as np
import pandas as pd  # Missing import
from urllib.parse import urlparse

# Load the dataset
input_file = "dataset/phishing_urls.csv"
output_file = "output/221IT047_URLfeaturedataset.csv"

df = pd.read_csv(input_file)

# Define phishing-related words
phishing_keywords = ['login', 'admin', 'secure', 'bank', 'account', 'verify', 'webscr', 'password']

def extract_features(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.netloc
    path = parsed_url.path

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": url.count('.'),
        "num_hyphens": url.count('-'),
        "num_slashes": url.count('/'),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(not c.isalnum() for c in url),
        "has_ip": int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", hostname))),
        "has_https": int(parsed_url.scheme == "https"),
        "num_subdomains": hostname.count('.') - 1,
        "contains_phishing_words": int(any(word in url.lower() for word in phishing_keywords)),
        "is_short_url": int(len(url) < 20),
        "is_tld_popular": int(tldextract.extract(url).suffix in ["com", "org", "net"]),
        "is_tld_suspicious": int(tldextract.extract(url).suffix in ["xyz", "top", "info", "tk", "ml"]),
        "has_at_symbol": int("@" in url),
        "has_double_slash": int("//" in url[7:]),  
        "has_redirect": int(">" in url or "<" in url),
        "has_suspicious_words": int(any(word in url.lower() for word in ["free", "win", "click", "offer"])),
        "ratio_digits": sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
        "ratio_special_chars": sum(not c.isalnum() for c in url) / len(url) if len(url) > 0 else 0,
        "is_long_url": int(len(url) > 75),
        "num_params": len(parsed_url.query.split('&')) if parsed_url.query else 0,
        "is_encoded": int("%" in url),
        "num_fragments": len(parsed_url.fragment.split('#')) if parsed_url.fragment else 0,
        "has_www": int("www." in hostname),
        "is_sensitive_tld": int(tldextract.extract(url).suffix in ["gov", "edu", "mil"]),
        "num_uppercase": sum(1 for c in url if c.isupper()),
        "num_lowercase": sum(1 for c in url if c.islower()),
        "entropy": -sum(p * np.log2(p) for p in np.bincount(np.frombuffer(url.encode(), dtype=np.uint8)) / len(url) if p > 0),
        "num_query_params": len(parsed_url.query.split("&")) if parsed_url.query else 0,
        "has_shortening_service": int(any(service in url for service in ["bit.ly", "tinyurl", "goo.gl"])),
        "num_encoded_chars": url.count('%'),
        "is_numeric_domain": int(hostname.replace(".", "").isdigit()),
        "num_subdirectory": path.count('/'),
        "has_js": int("javascript" in url.lower()),
        "has_mailto": int("mailto:" in url.lower()),
        "is_long_hostname": int(len(hostname) > 50),
        "is_long_path": int(len(path) > 50),
        "num_underscores": url.count('_'),
        "num_ampersands": url.count('&'),
        "has_equal_sign": int("=" in url),
        "num_percent_symbols": url.count('%'),
        "num_hash_symbols": url.count('#'),
        "is_query_long": int(len(parsed_url.query) > 50),
        "is_fragment_long": int(len(parsed_url.fragment) > 20),
        "contains_login_keywords": int(any(word in url.lower() for word in ["signin", "account", "password"])),
        "has_redirect_meta": int("<meta http-equiv=\"refresh\"" in url.lower()),
        "num_consecutive_dots": int(".." in url),
        "has_multiple_subdomains": int(hostname.count('.') > 2),
        "has_suspicious_chars": int(any(c in url for c in ['$', '^', '*', '|', '~'])),
        "has_tracking_params": int(any(param in url for param in ["utm_source", "utm_campaign"])),
        "is_url_too_long": int(len(url) > 100),
        "num_alphabets": sum(c.isalpha() for c in url),  # New Feature 1
        "is_punycode": int("xn--" in hostname)  # New Feature 2
    }

    #print(f"Extracted {len(features)} features for: {url}")  
    return list(features.values())  

# Apply feature extraction to all URLs
feature_df = df["URL"].apply(lambda x: extract_features(str(x))).apply(pd.Series)

# Add labels
feature_df["label"] = df["label"]

# Save the extracted features
feature_df.to_csv(output_file, index=False)
print(f"✅ Feature extraction complete! Dataset saved at {output_file}")
