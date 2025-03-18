import pandas as pd

# Load your dataset (modify the path accordingly)
df = pd.read_csv("output/221IT063-Pre-processed_Dataset.csv")  

# Convert all column names to lowercase to avoid case-sensitivity issues
df.columns = df.columns.str.lower()

# Define expected feature names (convert them to lowercase)
expected_features = [
    "urllength", "domainlength", "isdomainip", "urlsimilarityindex",
    "charcontinuationrate", "tldlegitimateprob", "urlcharprob", "tldlength",
    "noofsubdomain", "hasobfuscation", "noofobfuscatedchar", "obfuscationratio",
    "nooflettersinurl", "letterratioinurl", "noofdegitsinurl", "degitratioinurl",
    "noofequalsinurl", "noofqmarkinurl", "noofampersandinurl",
    "noofotherspecialcharsinurl", "spacialcharratioinurl", "ishttps",
    "lineofcode", "largestlinelength", "hastitle", "domaintitlematchscore",
    "urltitlematchscore", "hasfavicon", "robots", "isresponsive",
    "noofurlredirect", "noofselfredirect", "hasdescription", "noofpopup",
    "noofiframe", "hasexternalformsubmit", "hassocialnet", "hassubmitbutton",
    "hashiddenfields", "haspasswordfield", "bank", "pay", "crypto",
    "hascopyrightinfo", "noofimage", "noofcss", "noofjs", "noofselfref",
    "noofemptyref", "noofexternalref", "filename", "url", "domain", "tld", "title"
]

# Convert expected_features to lowercase
expected_features = [col.lower() for col in expected_features]

# Select only the expected features from the dataframe
df = df[expected_features]

print("✅ Features aligned successfully!")
