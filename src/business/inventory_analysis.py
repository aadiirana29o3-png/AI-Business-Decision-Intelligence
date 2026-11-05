
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Load original dataset
file_path = BASE_DIR / "data" / "raw" / "Online Retail.xlsx"
df = pd.read_excel(file_path)

# Clean the data
df = df.drop_duplicates()
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Calculate revenue
df["Sales"] = df["Quantity"] * df["UnitPrice"]

# Product-wise analysis
inventory = df.groupby(
    ["StockCode", "Description"]
).agg(
    Total_Quantity=("Quantity", "sum"),
    Total_Revenue=("Sales", "sum"),
    Total_Transactions=("InvoiceNo", "nunique")
).reset_index()

# Sort products by quantity sold
inventory = inventory.sort_values(
    by="Total_Quantity",
    ascending=False
)

# Save results
output_path = BASE_DIR / "data" / "processed" / "inventory_summary.csv"
inventory.to_csv(output_path, index=False)

print("Inventory analysis completed!")
print(inventory.head(10))
print("Total products:", len(inventory))
