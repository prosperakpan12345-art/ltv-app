import pandas as pd
from pathlib import Path

# =========================================================
# ELT TRANSFORM STAGE
# =========================================================

input_file = Path("data/raw/customer_data.csv")
output_file = Path("data/processed/customer_data_processed.csv")

print("=" * 60)
print("ELT TRANSFORM STAGE")
print("=" * 60)

# Automatically detect the separator
df = pd.read_csv(
    input_file,
    sep=None,
    engine="python"
)

print("\nRaw dataset shape:")
print(df.shape)

print("\nRaw columns:")
print(df.columns.tolist())

# Clean column names
df.columns = df.columns.str.strip()

# =========================================================
# CONVERT DATE
# =========================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# =========================================================
# CREATE MONTH AND SEASON
# =========================================================

df["Month"] = df["Date"].dt.month_name()

df["Season"] = "Q" + df["Date"].dt.quarter.astype("Int64").astype(str)

# =========================================================
# NUMERIC COLUMNS
# =========================================================

numeric_columns = [
    "CustomerID",
    "Age",
    "Income",
    "ProductsPurchased",
    "PurchaseFrequency",
    "AverageOrderValue",
    "SatisfactionScore",
    "CustomerLifetimeValue"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# =========================================================
# REMOVE DUPLICATES
# =========================================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\nDuplicates removed:", before - after)

# =========================================================
# MISSING VALUES
# =========================================================

print("\nMissing values:")
print(df.isnull().sum())

# =========================================================
# REORDER COLUMNS
# =========================================================

df = df[
    [
        "CustomerID",
        "Date",
        "Month",
        "Season",
        "Age",
        "Gender",
        "Region",
        "Income",
        "Membership",
        "ProductsPurchased",
        "PurchaseFrequency",
        "AverageOrderValue",
        "DiscountUsed",
        "MarketingChannel",
        "SatisfactionScore",
        "ChurnRisk",
        "CustomerLifetimeValue"
    ]
]

# =========================================================
# SAVE PROCESSED DATA
# =========================================================

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    output_file,
    index=False
)

# =========================================================
# FINAL REPORT
# =========================================================

print("\n" + "=" * 60)
print("TRANSFORMATION COMPLETE")
print("=" * 60)

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head().to_string(index=False))

print("\nMissing values after transformation:")
print(df.isnull().sum())

print("\nProcessed file saved to:")
print(output_file)