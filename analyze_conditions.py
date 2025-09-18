import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

def analyze_conditions():
    # Read the CSV file
    print("Reading dataset...")
    df = pd.read_csv('final_dataset_fixed.csv')
    
    # Count how many patients have more than 1 condition
    patients_with_multiple_conditions = (df['condition'].apply(lambda x: len(x.split('|')) > 1)).sum()
    print(f"Number of patients with more than 1 condition: {patients_with_multiple_conditions}")

    # Initialize a dictionary to count conditions
    condition_counts = {}
    condition_split_counts = {}
    
    # Count each condition's occurrence
    for conditions in df['condition']:
        # Split multiple conditions
        condition_list = conditions.split('|')
        # Count each individual condition
        for condition in condition_list:
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
    # Convert to pandas Series for easier handling
    condition_series = pd.Series(condition_counts)
    condition_series = condition_series.sort_values(ascending=False)
    
    # Calculate train/val/test splits (70:15:15)
    split_data = []
    print("\nCondition Distribution with Train/Val/Test Split (70:15:15):")
    print("-" * 80)
    
    # Create headers for the table
    headers = ["Condition", "Total", "Train (70%)", "Val (15%)", "Test (15%)", "% of Dataset"]
    
    total_samples = len(df)
    for condition, count in condition_series.items():
        train_count = int(count * 0.7)
        val_count = int(count * 0.15)
        test_count = count - train_count - val_count  # Ensure we account for rounding
        percentage = (count / total_samples) * 100
        
        split_data.append([
            condition,
            count,
            train_count,
            val_count,
            test_count,
            f"{percentage:.2f}%"
        ])
    
    # Print the table
    print(tabulate(split_data, headers=headers, tablefmt="grid"))
    
    # Create a bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=condition_series.values, y=condition_series.index)
    plt.title('Distribution of Conditions in Dataset')
    plt.xlabel('Number of Patients')
    plt.ylabel('Condition')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('condition_distribution.png')
    print("\nPlot saved as 'condition_distribution.png'")
    
    # Calculate imbalance metrics
    max_count = condition_series.max()
    min_count = condition_series.min()
    imbalance_ratio = max_count / min_count
    
    print("\nImbalance Analysis:")
    print("-" * 50)
    print(f"Total number of samples: {total_samples}")
    print(f"Most frequent condition: {condition_series.index[0]} ({max_count} samples)")
    print(f"Least frequent condition: {condition_series.index[-1]} ({min_count} samples)")
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")

if __name__ == "__main__":
    analyze_conditions()
