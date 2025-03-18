import pandas as pd

ROLL_NUMBER = "221IT047"

df = pd.read_csv("output/virus_total_results.csv")

# Identify constant columns (where all values are the same)
constant_columns = [col for col in df.columns if df[col].nunique() == 1]

# Print column numbers (indexes)
if constant_columns:
    print("Duplicate Columns (All values are the same):")
    for col in constant_columns:
        print(f"Column {df.columns.get_loc(col)}: {col}")

    # Save to output file
    output_filename = f"{ROLL_NUMBER}-Duplicate-Column.txt"
    with open(output_filename, "w") as f:
        f.write("\n".join([str(df.columns.get_loc(col)) for col in constant_columns]))

    print(f"Duplicate column indices saved to {output_filename}")
else:
    print("No duplicate columns found.")
