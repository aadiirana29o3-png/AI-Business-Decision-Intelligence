
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "product_daily_sales.csv"
MODEL_PATH = BASE_DIR / "src" / "models" / "product_demand_model.pkl"

# Load data and trained model
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
model = joblib.load(MODEL_PATH)

# Display available products
products = (
    df[["StockCode", "Description"]]
    .drop_duplicates()
    .sort_values("Description")
)

print("\n===== INVENTORY INTELLIGENCE SYSTEM =====")
print("\nAvailable Products:")
print(products.to_string(index=False))

# Product selection
product_code = input("\nEnter Product Code: ").strip()

if product_code not in df["StockCode"].astype(str).unique():
    print("Invalid product code.")
    raise SystemExit

# User-provided business information
current_stock = int(input("Enter current available stock (units): "))
lead_time = int(input("Enter supplier delivery time (days): "))
safety_days = int(input("Enter additional safety days: "))

if current_stock < 0 or lead_time < 1 or safety_days < 0:
    print("Invalid input. Check stock and number of days.")
    raise SystemExit

forecast_days = lead_time + safety_days

if forecast_days > 30:
    print("Total coverage period cannot exceed 30 days.")
    raise SystemExit

# Get selected product history
product_data = df[
    df["StockCode"].astype(str) == product_code
].sort_values("Date")

history = product_data["Quantity"].astype(float).tolist()
last_date = product_data["Date"].max()

description = product_data["Description"].iloc[0]

# Forecast demand
predictions = []

for i in range(1, forecast_days + 1):

    next_date = last_date + pd.Timedelta(days=i)

    input_data = pd.DataFrame([{
        "Day": next_date.day,
        "Month": next_date.month,
        "DayOfWeek": next_date.dayofweek,
        "PreviousDaySales": history[-1],
        "SalesLast7Days": np.mean(history[-7:])
    }])

    prediction = max(0, model.predict(input_data)[0])

    predictions.append(prediction)
    history.append(prediction)

# Calculate inventory requirement
expected_demand = sum(predictions)

suggested_reorder = max(
    0,
    int(np.ceil(expected_demand - current_stock))
)

# Results
print("\n===== INVENTORY ANALYSIS =====")
print("Product:", description)
print("Current Available Stock:", current_stock, "units")
print("Supplier Lead Time:", lead_time, "days")
print("Safety Period:", safety_days, "days")
print("Coverage Period:", forecast_days, "days")

print("\n===== AI DEMAND FORECAST =====")
print("Expected Demand:", round(expected_demand, 2), "units")

print("\n===== INVENTORY RECOMMENDATION =====")
print("Suggested Reorder Quantity:", suggested_reorder, "units")

if current_stock < expected_demand:
    print("Status: Potential Stock Shortage")
    print("Action: Review replenishment before the coverage period.")
else:
    print("Status: Stock appears sufficient for the estimated period.")
    print("Action: Monitor stock and review the next forecast.")

print("\nNote: Recommendation is based on estimated demand,")
print("user-entered stock and assumed supplier lead time.")
print("It does not account for existing purchase orders or uncertainty.")
