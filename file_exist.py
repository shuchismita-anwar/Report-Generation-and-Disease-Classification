import os
import pandas as pd

# Load the CSV
csv_path = 'trans-data.csv'
df = pd.read_csv(csv_path)

# Columns to check
columns_to_check = ['report_path', 'image_path', 'transcript_path']

# Track missing files
missing = {col: [] for col in columns_to_check}

for idx, row in df.iterrows():
    for col in columns_to_check:
        path = row[col]
        if not os.path.exists(path):
            missing[col].append((idx, path))

# Print summary
for col in columns_to_check:
    print(f"\nMissing files in column '{col}': {len(missing[col])}")
    for idx, path in missing[col][:10]:  # Show up to 10 missing paths per column
        print(f"  Row {idx}: {path}")
    if len(missing[col]) > 10:
        print(f"  ...and {len(missing[col])-10} more.")

print("\nFile existence check complete.")