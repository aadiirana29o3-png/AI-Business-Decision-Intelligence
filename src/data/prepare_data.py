import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Load original dataset
raw_path = BASE_DIR / "data" / "raw" / "Online Retail.xlsx"
df = pd.read_excel(raw_path)

# Remove duplicate records
df = df.drop_duplicates()

# Remove cancelled transactions
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Keep only valid quantities and prices
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Save cleaned dataset
clean_path = BASE_DIR / "data" / "processed" / "cleaned_retail.xlsx"
df.to_excel(clean_path, index=False)

# Calculate sales
df["Sales"] = df["Quantity"] * df["UnitPrice"]

# Calculate daily sales
daily_sales = df.groupby(
    df["InvoiceDate"].dt.date
)["Sales"].sum().reset_index()

daily_sales.columns = ["Date", "Sales"]

# Save daily sales
daily_path = BASE_DIR / "data" / "processed" / "daily_sales.csv"
daily_sales.to_csv(daily_path, index=False)

print("Data preparation completed!")
print("Cleaned records:", len(df))
print(daily_sales.head())
