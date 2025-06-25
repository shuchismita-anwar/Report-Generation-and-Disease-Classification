import pandas as pd

# Load test dataset
df = pd.read_csv('../../dataset_splits/test.csv')
print(f"Test dataset has {len(df)} rows")

# Check specific IDs from our LLM reports
test_ids = [
    '002da0d9-ce49c30d-4dfcc1f8-746d2401-d8044d48',
    '005aff0f-0c236062-06df954a-25ad1874-bcdffcb0', 
    '0066734a-35568fde-fd52ba23-ec66f3de-88d4aaf9'
]

print(f"\nChecking for {len(test_ids)} LLM report IDs in test dataset...")

matches = df[df['dicom_id'].isin(test_ids)]
print(f"Found {len(matches)} matches in test split")

if len(matches) > 0:
    print("\nMatched entries:")
    for _, row in matches.iterrows():
        print(f"  {row['dicom_id']} -> {row['report_path']}")
else:
    print("None of the LLM report IDs found in test split")
    
    # Check other splits
    for split in ['train', 'val']:
        try:
            split_df = pd.read_csv(f'../../dataset_splits/{split}.csv')
            split_matches = split_df[split_df['dicom_id'].isin(test_ids)]
            print(f"Found {len(split_matches)} matches in {split} split")
        except FileNotFoundError:
            print(f"{split}.csv not found") 