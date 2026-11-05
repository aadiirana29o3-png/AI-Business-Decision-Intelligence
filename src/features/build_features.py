import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Load complete daily sales data
file_path = BASE_DIR / "data" / "processed" / "daily_sales_complete.csv"
df = pd.read_csv(file_path, parse_dates=["Date"])

# Date-based features
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Previous day sales
df["PreviousDaySales"] = df["Sales"].shift(1)

# Previous 7 days average sales
df["SalesLast7Days"] = df["Sales"].shift(1).rolling(7).mean()

# Remove rows with missing feature values
df = df.dropna()

# Save feature dataset
output_path = BASE_DIR / "data" / "processed" / "sales_features.csv"
df.to_csv(output_path, index=False)

print("Feature engineering completed!")
print(df.head())
print("Total records:", len(df))
