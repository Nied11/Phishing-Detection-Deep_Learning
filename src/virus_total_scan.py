import requests
import pandas as pd
import time
from tqdm import tqdm

# 🔹 Replace with your actual VirusTotal API Key
API_KEY = "7afd056069e36961b3d3773366f2b97a04cae5dae22e402271216a2989668ec2"

# 🔹 File paths
INPUT_FILE = "dataset/phishing_urls.csv"  # Original dataset
OUTPUT_FILE = "output/virus_total_results.csv"  # Updated dataset

# 🔹 VirusTotal API URL
VT_SCAN_URL = "https://www.virustotal.com/api/v3/urls"
VT_REPORT_URL = "https://www.virustotal.com/api/v3/analyses/"

# 🔹 Headers for API requests
HEADERS = {"x-apikey": API_KEY}

# 🔹 Function to submit a URL for scanning
def scan_url(url):
    try:
        response = requests.post(VT_SCAN_URL, headers=HEADERS, data={"url": url})
        if response.status_code == 200:
            return response.json().get("data", {}).get("id")
        else:
            print(f"❌ Error scanning {url}: {response.json()}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

# 🔹 Function to get scan results
def get_scan_results(scan_id):
    try:
        response = requests.get(f"{VT_REPORT_URL}{scan_id}", headers=HEADERS)
        if response.status_code == 200:
            analysis = response.json().get("data", {}).get("attributes", {}).get("stats", {})
            # If malicious detections exist, classify as "phishing", otherwise "benign"
            if analysis.get("malicious", 0) > 0:
                return "phishing"
            else:
                return "benign"
        else:
            return "unknown"
    except Exception as e:
        print(f"❌ Error fetching scan results: {e}")
        return "unknown"

# 🔹 Load dataset and check columns
df = pd.read_csv(INPUT_FILE)
if "label" not in df.columns:
    df["label"] = "unknown"  # Default label if missing

# 🔹 Process first 500 URLs
for index, row in tqdm(df.iloc[:500].iterrows(), total=500):
    url = row["URL"]
    print(f"🔍 Scanning URL {index + 1}: {url}")

    scan_id = scan_url(url)
    if scan_id:
        time.sleep(20)  # Wait to avoid API rate limits

        df["label"] = df["label"].astype(str)  # Ensure column is of type string
        df.at[index, "label"] = get_scan_results(scan_id)


# 🔹 Save updated dataset
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Updated dataset saved as {OUTPUT_FILE}")
