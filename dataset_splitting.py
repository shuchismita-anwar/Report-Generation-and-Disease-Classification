import pandas as pd
import numpy as np
import random
from collections import Counter
import os

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Read the dataset
print("Reading dataset...")
df = pd.read_csv('final_dataset_fixed.csv')

print(f"Total records: {len(df)}")
print(f"Unique conditions: {df['condition'].nunique()}")

# Manual split that maintains condition distribution
# Step 1: Group by conditions
condition_groups = {}
for idx, row in df.iterrows():
    condition = row['condition']
    if condition not in condition_groups:
        condition_groups[condition] = []
    condition_groups[condition].append(idx)

# Print condition distribution
print("\nCondition distribution (top 10):")
condition_counts = [(condition, len(indices)) for condition, indices in condition_groups.items()]
condition_counts.sort(key=lambda x: x[1], reverse=True)
for condition, count in condition_counts[:10]:
    print(f"{condition}: {count}")

# Step 2: Allocate indices to train, test, and validation sets
train_indices = []
test_indices = []
val_indices = []

# Process each condition group to maintain stratification
for condition, indices in condition_groups.items():
    # Shuffle indices to ensure randomness
    shuffled_indices = indices.copy()
    random.shuffle(shuffled_indices)
    
    # Calculate split sizes
    n_samples = len(shuffled_indices)
    n_train = int(n_samples * 0.7)
    n_test = int(n_samples * 0.15)
    n_val = n_samples - n_train - n_test
    
    # Assign indices to sets
    if n_samples < 3:
        # For rare conditions with very few samples, prioritize training
        if n_samples == 1:
            train_indices.extend(shuffled_indices)
        elif n_samples == 2:
            train_indices.append(shuffled_indices[0])
            test_indices.append(shuffled_indices[1])
    else:
        # For conditions with sufficient samples, use the 70:15:15 split
        train_indices.extend(shuffled_indices[:n_train])
        test_indices.extend(shuffled_indices[n_train:n_train+n_test])
        val_indices.extend(shuffled_indices[n_train+n_test:])

# Create dataframes for each split
train_df = df.loc[train_indices].copy()
test_df = df.loc[test_indices].copy()
val_df = df.loc[val_indices].copy()

# Verify the splits
print(f"\nTrain set: {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
print(f"Test set: {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")
print(f"Validation set: {len(val_df)} records ({len(val_df)/len(df)*100:.1f}%)")

# Check for any duplicate indices across splits
train_set = set(train_indices)
test_set = set(test_indices)
val_set = set(val_indices)

train_test_overlap = train_set.intersection(test_set)
train_val_overlap = train_set.intersection(val_set)
test_val_overlap = test_set.intersection(val_set)

if train_test_overlap or train_val_overlap or test_val_overlap:
    print("WARNING: Overlapping indices detected between splits!")
else:
    print("No overlapping indices between splits.")

# Check if all samples are accounted for
total_after_split = len(train_df) + len(test_df) + len(val_df)
if total_after_split == len(df):
    print(f"All samples accounted for ({total_after_split}/{len(df)})")
else:
    print(f"WARNING: Sample count mismatch: {total_after_split} vs {len(df)}")

# Verify condition distribution in splits
train_conditions = Counter(train_df['condition'])
test_conditions = Counter(test_df['condition'])
val_conditions = Counter(val_df['condition'])

print("\nCondition distribution in splits (top 10):")
condition_stats = []
for condition in [x[0] for x in condition_counts[:10]]:
    total = condition_groups[condition]
    train = train_conditions.get(condition, 0)
    test = test_conditions.get(condition, 0)
    val = val_conditions.get(condition, 0)
    train_pct = train / len(total) * 100 if total else 0
    test_pct = test / len(total) * 100 if total else 0
    val_pct = val / len(total) * 100 if total else 0
    
    print(f"{condition}: Total={len(total)}, Train={train} ({train_pct:.1f}%), Test={test} ({test_pct:.1f}%), Val={val} ({val_pct:.1f}%)")

# Print rare conditions distribution
print("\nDistribution of rare conditions (count <= 5):")
rare_conditions = [condition for condition, count in condition_counts if count <= 5]
for condition in rare_conditions[:10]:  # Show top 10 rare conditions
    total = condition_groups[condition]
    train = train_conditions.get(condition, 0)
    test = test_conditions.get(condition, 0)
    val = val_conditions.get(condition, 0)
    print(f"{condition}: Total={len(total)}, Train={train}, Test={test}, Val={val}")

# Save the split datasets
print("\nSaving splits to CSV files...")
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
val_df.to_csv('val.csv', index=False)

print("Dataset splitting complete.")

# Print file sizes
print("\nVerifying files created:")
for file in ['train.csv', 'test.csv', 'val.csv']:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"{file}: {size_mb:.2f} MB, {len(pd.read_csv(file))} records")
    else:
        print(f"{file} not found!")

# Analyze individual condition distributions (handling multi-label cases)
print("\n" + "="*80)
print("DETAILED CONDITION DISTRIBUTION ANALYSIS")
print("="*80)

# Define the specific conditions we're interested in
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Edema',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pneumonia', 'Support Devices'
]

# Function to count individual conditions in a dataset
def count_individual_conditions(dataset):
    condition_counts = {condition: 0 for condition in CONDITIONS}
    
    for _, row in dataset.iterrows():
        # Split combined conditions into individual ones
        conditions = row['condition'].split('|')
        for condition in conditions:
            if condition in CONDITIONS:
                condition_counts[condition] += 1
    
    return condition_counts

# Count conditions in each split
train_individual = count_individual_conditions(train_df)
test_individual = count_individual_conditions(test_df)
val_individual = count_individual_conditions(val_df)
total_individual = count_individual_conditions(df)

# Print distribution table
print("\nIndividual Condition Distribution:")
print(f"{'Condition':<20} {'Total':<10} {'Train':<15} {'Test':<15} {'Val':<15}")
print("-" * 80)

for condition in CONDITIONS:
    total = total_individual[condition]
    train = train_individual[condition]
    test = test_individual[condition]
    val = val_individual[condition]
    
    train_pct = train / total * 100 if total else 0
    test_pct = test / total * 100 if total else 0
    val_pct = val / total * 100 if total else 0
    
    print(f"{condition:<20} {total:<10} {train:<6} ({train_pct:.1f}%) {test:<6} ({test_pct:.1f}%) {val:<6} ({val_pct:.1f}%)")

# Calculate frequency of multi-label samples
multi_label_total = sum(1 for _, row in df.iterrows() if '|' in row['condition'])
multi_label_train = sum(1 for _, row in train_df.iterrows() if '|' in row['condition'])
multi_label_test = sum(1 for _, row in test_df.iterrows() if '|' in row['condition'])
multi_label_val = sum(1 for _, row in val_df.iterrows() if '|' in row['condition'])

print("\nMulti-label samples:")
print(f"Total: {multi_label_total} ({multi_label_total/len(df)*100:.1f}%)")
print(f"Train: {multi_label_train} ({multi_label_train/len(train_df)*100:.1f}%)")
print(f"Test: {multi_label_test} ({multi_label_test/len(test_df)*100:.1f}%)")
print(f"Val: {multi_label_val} ({multi_label_val/len(val_df)*100:.1f}%)")
