import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parents[2]

# Load daily sales data
file_path = BASE_DIR / "data" / "processed" / "daily_sales.csv"
df = pd.read_csv(file_path, parse_dates=["Date"])

# Create sales graph
plt.figure(figsize=(14, 6))
plt.plot(df["Date"], df["Sales"])

plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales (£)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save graph
output_path = BASE_DIR / "reports" / "daily_sales_trend.png"
plt.savefig(output_path)

plt.show()

print("Sales graph generated successfully!")
