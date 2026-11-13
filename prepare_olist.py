
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "external"

# Load datasets
orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
translations = pd.read_csv(
    DATA_DIR / "product_category_name_translation.csv"
)

print("All 4 datasets loaded successfully.")

# Keep delivered orders
orders = orders[
    orders["order_status"] == "delivered"
].copy()

# Combine order items and orders
df = items.merge(
    orders[["order_id", "order_purchase_timestamp"]],
    on="order_id",
    how="inner"
)

# Add product information
df = df.merge(
    products[["product_id", "product_category_name"]],
    on="product_id",
    how="left"
)

# Translate Portuguese categories into English
df = df.merge(
    translations,
    on="product_category_name",
    how="left"
)

# Use English category, Portuguese fallback, or Uncategorized
df["Category"] = (
    df["product_category_name_english"]
    .fillna(df["product_category_name"])
    .fillna("Uncategorized")
)

# Prepare date
df["Date"] = pd.to_datetime(
    df["order_purchase_timestamp"],
    errors="coerce"
)

# Create shorter product labels using product ID
short_id = df["product_id"].astype(str).str[:8]

df["Product"] = df["Category"] + " | " + short_id

# If short IDs collide, use full IDs for those products
duplicate_labels = df["Product"].duplicated(keep=False)

df.loc[duplicate_labels, "Product"] = (
    df.loc[duplicate_labels, "Category"]
    + " | "
    + df.loc[duplicate_labels, "product_id"].astype(str)
)

# Each order-item row represents one purchased unit
df["Quantity"] = 1

df["UnitPrice"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

# Remove invalid records
df = df.dropna(
    subset=["Date", "Product", "UnitPrice"]
)

df = df[df["UnitPrice"] > 0]

# Final dashboard-compatible dataset
final_df = df[
    ["Date", "Product", "Quantity", "UnitPrice"]
].copy()

# Save output
output_path = DATA_DIR / "olist_prepared.csv"

final_df.to_csv(output_path, index=False)

print("\nData preparation completed!")
print("Total records:", len(final_df))
print("Unique products:", final_df["Product"].nunique())
print("Date range:", final_df["Date"].min(), "to", final_df["Date"].max())
print("Saved file:", output_path)
