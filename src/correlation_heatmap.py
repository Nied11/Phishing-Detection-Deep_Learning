import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("dataset\phishing_urls.csv")

# Convert all applicable columns to numeric, coercing errors
df_numeric = df.apply(pd.to_numeric, errors="coerce")

# Drop any non-numeric columns that couldn't be converted
df_numeric = df_numeric.dropna(axis=1, how="all")  # Drops columns where all values are NaN

# Compute the correlation matrix
correlation_matrix = df_numeric.corr()

# Set figure size for better readability
plt.figure(figsize=(12, 8))

# Generate heatmap with improved clarity
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    annot_kws={"size": 6},
    linewidths=0.5,
    linecolor="black"
)

plt.title("Feature Correlation Heatmap", fontsize=14)

plt.savefig("output/221IT047-Heatmap.JPEG", dpi=300, bbox_inches="tight")

plt.show()



