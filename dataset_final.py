import pandas as pd
import os

# Define file paths
dataset_path = "final_dataset.csv"
output_path = "final_dataset_fixed.csv"

# List of unwanted conditions (from condition_removal_check.py)
unwanted_conditions = [
    "Consolidation",
    "Pneumothorax",
    "Lung Lesion",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Pleural Other",
]

# Function to check if any unwanted condition is present in the condition string
# (from condition_removal_check.py)
def contains_unwanted(condition_str):
    if pd.isna(condition_str):
        return False
    return any(cond in condition_str.split('|') for cond in unwanted_conditions)

# Read the dataset
df = pd.read_csv(dataset_path)

# Remove the 'reflax_id' column
df = df.drop(columns=['reflax_id'])

# Drop rows with any missing values
df = df.dropna()

# Remove rows with unwanted conditions (from condition_removal_check.py)
original_shape = df.shape
rows_to_remove = df['condition'].apply(contains_unwanted).sum()
df = df[~df['condition'].apply(contains_unwanted)].reset_index(drop=True)

# Save the cleaned dataset
df.to_csv(output_path, index=False)

print(f"Dataset processed successfully!")
print(f"Original shape: {pd.read_csv(dataset_path).shape}")
print(f"After cleaning: {df.shape}")
print(f"Rows removed due to unwanted conditions: {rows_to_remove}")
print(f"Saved as: {output_path}")