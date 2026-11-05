
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "product_daily_sales.csv"
MODEL_PATH = BASE_DIR / "src" / "models" / "product_demand_model.pkl"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "selected_product_forecast.csv"

# Load data and trained model
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
model = joblib.load(MODEL_PATH)

# Show available products
products = (
    df[["StockCode", "Description"]]
    .drop_duplicates()
    .sort_values("Description")
)

print("\n===== AVAILABLE PRODUCTS =====")
print(products.to_string(index=False))

# Get user selection
product_code = input("\nEnter Product Code: ").strip()

if product_code not in df["StockCode"].astype(str).unique():
    print("Invalid product code. Please try again.")
    raise SystemExit

days = int(input("Enter forecast days (1-30): "))

if days < 1 or days > 30:
    print("Please enter days between 1 and 30.")
    raise SystemExit

# Get selected product history
product_data = df[
    df["StockCode"].astype(str) == product_code
].sort_values("Date").copy()

history = product_data["Quantity"].astype(float).tolist()
last_date = product_data["Date"].max()

description = product_data["Description"].iloc[0]

forecast_results = []

# Predict future demand one day at a time
for i in range(1, days + 1):

    next_date = last_date + pd.Timedelta(days=i)

    previous_day_sales = history[-1]
    last_7_days_average = np.mean(history[-7:])

    input_data = pd.DataFrame([{
        "Day": next_date.day,
        "Month": next_date.month,
        "DayOfWeek": next_date.dayofweek,
        "PreviousDaySales": previous_day_sales,
        "SalesLast7Days": last_7_days_average
    }])

    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)

    forecast_results.append({
        "Date": next_date,
        "StockCode": product_code,
        "Description": description,
        "PredictedDemand": round(prediction, 2)
    })

    # Add prediction to history for next day's forecast
    history.append(prediction)

# Save forecast
forecast_df = pd.DataFrame(forecast_results)
forecast_df.to_csv(OUTPUT_PATH, index=False)

print("\n===== PRODUCT DEMAND FORECAST =====")
print(forecast_df.to_string(index=False))

print("\nTotal predicted demand:",
      round(forecast_df["PredictedDemand"].sum(), 2), "units")

print("\nForecast saved successfully!")
