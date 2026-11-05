import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Load inventory data
file_path = BASE_DIR / "data" / "processed" / "inventory_summary.csv"
df = pd.read_csv(file_path)

# Select top 10 products
top_products = df.nlargest(10, "Total_Quantity").copy()

# Create product labels
top_products["Product"] = (
    top_products["StockCode"].astype(str)
    + " - "
    + top_products["Description"].str.slice(0, 25)
)

# Create bar chart
plt.figure(figsize=(12, 7))

plt.barh(
    top_products["Product"],
    top_products["Total_Quantity"]
)

plt.title("Top 10 Products by Quantity Sold")
plt.xlabel("Total Quantity Sold")
plt.ylabel("Products")
plt.gca().invert_yaxis()

plt.tight_layout()

# Save graph
output_path = BASE_DIR / "reports" / "top_inventory_products.png"
plt.savefig(output_path)
plt.show()

print("Inventory visualization completed!")