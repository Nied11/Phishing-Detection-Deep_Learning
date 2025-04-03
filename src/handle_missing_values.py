import pandas as pd

df = pd.read_csv("dataset\extracted_features_normalized.csv")

# Identify missing values
missing_values = df.isnull().sum()
missing_cols = missing_values[missing_values > 0].index.tolist()

if not missing_cols:
    print("No missing values found in the dataset.")
else:
    print("Missing values found in columns:", missing_cols)

    # Fill missing values
    for col in missing_cols:
        if df[col].dtype in ['int64', 'float64']:  # Numerical column
            df[col].fillna(df[col].median(), inplace=True)  # Median for robustness
        else:  # Categorical column
            df[col].fillna(df[col].mode()[0], inplace=True)  # Most frequent value
    
    # Save the cleaned dataset
    df.to_csv("dataset\extracted_features_filled.csv", index=False)
    
    print("Missing values handled. Updated dataset saved as extracted_features_filled.csv.")
