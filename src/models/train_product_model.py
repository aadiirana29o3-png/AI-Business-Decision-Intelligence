
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[2]

# Load product-wise sales data
file_path = BASE_DIR / "data" / "processed" / "product_daily_sales.csv"
df = pd.read_csv(file_path, parse_dates=["Date"])

# Sort data product-wise and date-wise
df = df.sort_values(["StockCode", "Date"]).copy()

# Create features separately for each product
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

# Split chronologically within each product
train_parts = []
test_parts = []

for code, product_data in df.groupby("StockCode"):
    split = int(len(product_data) * 0.8)

    train_parts.append(product_data.iloc[:split])
    test_parts.append(product_data.iloc[split:])

train_data = pd.concat(train_parts)
test_data = pd.concat(test_parts)

X_train = train_data[features]
y_train = train_data["Quantity"]

X_test = test_data[features]
y_test = test_data["Quantity"]

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
predictions = np.maximum(predictions, 0)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("===== PRODUCT DEMAND MODEL =====")
print("Training records:", len(train_data))
print("Testing records:", len(test_data))
print("Products:", df["StockCode"].nunique())
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Save model
model_path = BASE_DIR / "src" / "models" / "product_demand_model.pkl"
joblib.dump(model, model_path)

# Save feature information
feature_path = BASE_DIR / "src" / "models" / "product_model_features.txt"
feature_path.write_text("\n".join(features))

print("\nProduct demand model saved successfully!")
