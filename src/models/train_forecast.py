
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[2]

# Load feature dataset
file_path = BASE_DIR / "data" / "processed" / "sales_features.csv"
df = pd.read_csv(file_path, parse_dates=["Date"])

# Select features and target
features = [
    "Day",
    "Month",
    "DayOfWeek",
    "PreviousDaySales",
    "SalesLast7Days"
]

X = df[features]
y = df["Sales"]

# Chronological train-test split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Baseline predictions
baseline_predictions = X_test["PreviousDaySales"]

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
model_mae = mean_absolute_error(y_test, predictions)

baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
model_rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Baseline Model MAE:", round(baseline_mae, 2))
print("Random Forest MAE:", round(model_mae, 2))
print("Baseline Model RMSE:", round(baseline_rmse, 2))
print("Random Forest RMSE:", round(model_rmse, 2))

# Save trained model
model_path = BASE_DIR / "src" / "models" / "sales_forecast_model.pkl"
joblib.dump(model, model_path)

# Save forecast results
forecast_results = pd.DataFrame({
    "Date": df["Date"].iloc[split:].values,
    "ActualSales": y_test.values,
    "PredictedSales": predictions
})

results_path = BASE_DIR / "data" / "processed" / "forecast_results.csv"
forecast_results.to_csv(results_path, index=False)

print("Model saved successfully!")
print("Forecast results saved successfully!")
