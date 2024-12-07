import pandas as pd

# Define clinical columns
clinical_cols = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "acuity",
    "age",
]

# Read the test.csv file
df = pd.read_csv("test22222.csv")

# Calculate missing values for each column
missing_values = {}
for col in clinical_cols:
    if col in df.columns:
        missing_count = df[col].isna().sum()
        total_count = len(df)
        missing_percentage = (missing_count / total_count) * 100
        missing_values[col] = {
            "missing_count": missing_count,
            "total_count": total_count,
            "missing_percentage": missing_percentage
        }
    else:
        missing_values[col] = {
            "missing_count": "Column not found",
            "total_count": len(df),
            "missing_percentage": "N/A"
        }

# Print results
print("\nMissing Values Analysis:")
print("-" * 60)
print(f"{'Column':<15} {'Missing':<10} {'Total':<10} {'Percentage':<10}")
print("-" * 60)
for col, stats in missing_values.items():
    if isinstance(stats["missing_count"], str):
        print(f"{col:<15} {stats['missing_count']:<10} {stats['total_count']:<10} {'N/A':<10}")
    else:
        print(f"{col:<15} {stats['missing_count']:<10} {stats['total_count']:<10} {stats['missing_percentage']:.2f}%")
