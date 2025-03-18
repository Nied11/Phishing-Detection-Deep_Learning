import pandas as pd

ROLL_NUMBER = "221IT063"

df = pd.read_csv("output/virus_total_results.csv")

# Identify duplicate rows (excluding the index column)
duplicate_rows = df[df.duplicated()].index.tolist()

# Print duplicate row numbers
if duplicate_rows:
    print("Duplicate Rows Found:")
    for row in duplicate_rows:
        print(f"Row {row}")

    # Save to output file
    output_filename = f"{ROLL_NUMBER}-Duplicate-Row.txt"
    with open(output_filename, "w") as f:
        f.write("\n".join(map(str, duplicate_rows)))

    print(f"Duplicate row indices saved to {output_filename}")
else:
    print("No duplicate rows found.")
