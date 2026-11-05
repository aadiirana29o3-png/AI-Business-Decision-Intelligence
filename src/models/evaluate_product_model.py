
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[2]

# Load data and trained model
data_path = BASE_DIR / "data" / "processed" / "product_daily_sales.csv"
model_path = BASE_DIR / "src" / "models" / "product_demand_model.pkl"

df = pd.read_csv(data_path, parse_dates=["Date"])
model = joblib.load(model_path)

# Recreate the same features used during training
df = df.sort_values(["StockCode", "Date"]).copy()

group = df.groupby("StockCode")["Quantity"]

df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek

df["PreviousDaySales"] = group.shift(1)
df["SalesLast7Days"] = group.transform(
    lambda x: x.shift(1).rolling(7).mean()
)

df = df.dropna().copy()

features = [
    "Day",
    "Month",
    "DayOfWeek",
    "PreviousDaySales",
    "SalesLast7Days"
]

# Recreate chronological test split
test_parts = []

for code, product_data in df.groupby("StockCode"):
    split = int(len(product_data) * 0.8)
    test_parts.append(product_data.iloc[split:])

test_data = pd.concat(test_parts).copy()

# Predict demand
test_data["PredictedQuantity"] = model.predict(
    test_data[features]
)

test_data["PredictedQuantity"] = (
    test_data["PredictedQuantity"].clip(lower=0)
)

# Calculate product-wise performance
results = []

for code, product_data in test_data.groupby("StockCode"):

    actual = product_data["Quantity"]
    predicted = product_data["PredictedQuantity"]
    baseline = product_data["PreviousDaySales"]

    results.append({
        "StockCode": code,
        "Description": product_data["Description"].iloc[0],
        "TestRecords": len(product_data),
        "Model_MAE": mean_absolute_error(actual, predicted),
        "Baseline_MAE": mean_absolute_error(actual, baseline)
    })

result_df = pd.DataFrame(results)

result_df["Improvement_Percent"] = (
    (result_df["Baseline_MAE"] - result_df["Model_MAE"])
    / result_df["Baseline_MAE"].replace(0, np.nan)
) * 100

result_df = result_df.sort_values("Model_MAE")

# Save results
output_path = (
    BASE_DIR / "data" / "processed" / "product_model_evaluation.csv"
)

result_df.to_csv(output_path, index=False)

print("===== PRODUCT-WISE MODEL EVALUATION =====")
print(result_df.round(2).to_string(index=False))

print("\nEvaluation saved successfully!")

