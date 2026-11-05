
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Load cleaned retail data and inventory summary
retail_path = BASE_DIR / "data" / "processed" / "cleaned_retail.xlsx"
inventory_path = BASE_DIR / "data" / "processed" / "inventory_summary.csv"

df = pd.read_excel(retail_path)
inventory = pd.read_csv(inventory_path)

# Standardize product codes
df["StockCode"] = df["StockCode"].astype(str).str.strip()
inventory["StockCode"] = inventory["StockCode"].astype(str).str.strip()

# Select 20 frequently purchased products
top_products = (
    inventory.sort_values("Total_Transactions", ascending=False)
    .head(20)
)

selected_codes = top_products["StockCode"].tolist()

# Keep selected products
df = df[df["StockCode"].isin(selected_codes)].copy()

# Prepare dates
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Date"] = df["InvoiceDate"].dt.normalize()

# Aggregate daily quantity for each product
daily = (
    df.groupby(["StockCode", "Date"])["Quantity"]
    .sum()
    .reset_index()
)

# Create a complete daily date range
all_dates = pd.date_range(
    start=daily["Date"].min(),
    end=daily["Date"].max(),
    freq="D"
)

# Fill missing product dates with zero sales
all_products = []

for code in selected_codes:
    product_data = daily[daily["StockCode"] == code]

    if product_data.empty:
        continue

    product_data = (
        product_data.set_index("Date")["Quantity"]
        .reindex(all_dates, fill_value=0)
        .rename("Quantity")
        .reset_index()
        .rename(columns={"index": "Date"})
    )

    product_data["StockCode"] = code
    all_products.append(product_data)

# Combine all products
result = pd.concat(all_products, ignore_index=True)

# Add product descriptions
descriptions = (
    df.groupby("StockCode")["Description"]
    .agg(lambda x: x.dropna().mode().iloc[0] if not x.dropna().mode().empty else "Unknown")
)

result["Description"] = result["StockCode"].map(descriptions)

# Arrange columns
result = result[
    ["Date", "StockCode", "Description", "Quantity"]
]

# Save dataset
output_path = BASE_DIR / "data" / "processed" / "product_daily_sales.csv"
result.to_csv(output_path, index=False)

print("Product-wise dataset created successfully!")
print("Selected products:", result["StockCode"].nunique())
print("Total records:", len(result))
print("\nSample data:")
print(result.head(10))
