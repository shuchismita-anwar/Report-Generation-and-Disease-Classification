import pandas as pd

# File path
csv_path = "final_dataset_fixed.csv"
output_path = "final_dataset_fixed.csv"

# List of unwanted conditions
unwanted_conditions = [
    "Consolidation",
    "Pneumothorax",
    "Lung Lesion",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Pleural Other",
]

# Load dataset
df = pd.read_csv(csv_path)

# Function to check if any unwanted condition is present in the condition string
def contains_unwanted(condition_str):
    if pd.isna(condition_str):
        return False
    return any(cond in condition_str.split('|') for cond in unwanted_conditions)

# Apply filter
total_rows = len(df)
rows_to_remove = df['condition'].apply(contains_unwanted).sum()

print(f"Total rows in dataset: {total_rows}")
print(f"Rows that would be removed: {rows_to_remove}")
print(f"Rows remaining after removal: {total_rows - rows_to_remove}")

# Remove rows with unwanted conditions
df_clean = df[~df['condition'].apply(contains_unwanted)].reset_index(drop=True)

print(f"\nRemoved rows with unwanted conditions. New shape: {df_clean.shape}")

# Save the cleaned dataframe
df_clean.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")