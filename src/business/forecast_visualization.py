import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parents[2]

# Load dataset
file_path = BASE_DIR / "data" / "processed" / "sales_features.csv"
df = pd.read_csv(file_path, parse_dates=["Date"])

features = [
    "Day", "Month", "DayOfWeek",
    "PreviousDaySales", "SalesLast7Days"
]

X = df[features]
y = df["Sales"]

# Same chronological split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Compare actual and predicted sales
plt.figure(figsize=(14, 6))
plt.plot(df["Date"].iloc[split:], y_test, label="Actual Sales")
plt.plot(df["Date"].iloc[split:], predictions, label="Predicted Sales")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales (£)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save graph
output_path = BASE_DIR / "reports" / "actual_vs_predicted.png"
plt.savefig(output_path)
plt.show()

print("Forecast visualization generated successfully!")