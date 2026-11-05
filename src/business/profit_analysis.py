
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Load cleaned transaction data
file_path = BASE_DIR / "data" / "raw" / "Online Retail.xlsx"
df = pd.read_excel(file_path)

# Clean data
df = df.drop_duplicates()
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Calculate revenue
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Product-wise revenue analysis
profit_data = df.groupby(
    ["StockCode", "Description"]
).agg(
    Total_Quantity=("Quantity", "sum"),
    Total_Revenue=("Revenue", "sum"),
    Average_Selling_Price=("UnitPrice", "mean")
).reset_index()

# Sort by revenue
profit_data = profit_data.sort_values(
    by="Total_Revenue",
    ascending=False
)

# Save results
output_path = BASE_DIR / "data" / "processed" / "profit_analysis.csv"
profit_data.to_csv(output_path, index=False)

print("Revenue analysis completed!")
print(profit_data.head(10))
