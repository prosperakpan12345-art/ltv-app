import pandas as pd
from pathlib import Path

# Path to the raw customer dataset
file_path = Path("data/raw/customer_data.csv")

# Check if the dataset exists
if not file_path.exists():
    print("ERROR: customer_data.csv was not found.")
    print(f"Expected location: {file_path}")
    exit()

# Load the dataset
df = pd.read_csv(file_path, sep="\t")

df.columns = df.columns.str.strip()

print("=" * 60)
print("CUSTOMER LIFETIME VALUE DATASET - 2026")
print("=" * 60)
 
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())