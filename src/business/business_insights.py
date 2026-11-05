
import pandas as pd
import numpy as np

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

# Load business datasets
sales = pd.read_csv(
    DATA_DIR / "product_daily_sales.csv",
    parse_dates=["Date"]
)

revenue = pd.read_csv(DATA_DIR / "profit_analysis.csv")
inventory = pd.read_csv(DATA_DIR / "inventory_summary.csv")

print("\n===== AI BUSINESS INSIGHTS ENGINE =====")

# 1. Revenue insights
total_revenue = revenue["Total_Revenue"].sum()

top_revenue_product = revenue.loc[
    revenue["Total_Revenue"].idxmax()
]

print("\n1. REVENUE INSIGHT")
print("Total Revenue: £", round(total_revenue, 2))
print(
    "Highest Revenue Product:",
    top_revenue_product["Description"]
)
print(
    "Product Revenue: £",
    round(top_revenue_product["Total_Revenue"], 2)
)

# 2. Demand insights
demand_summary = (
    sales.groupby(["StockCode", "Description"])
    .agg(
        Total_Units=("Quantity", "sum"),
        Average_Daily_Demand=("Quantity", "mean"),
        Demand_Variation=("Quantity", "std")
    )
    .reset_index()
)

highest_demand = demand_summary.loc[
    demand_summary["Total_Units"].idxmax()
]

print("\n2. DEMAND INSIGHT")
print("Highest Quantity Sold:", highest_demand["Description"])
print("Total Units Sold:", int(highest_demand["Total_Units"]))

# 3. Demand consistency insight
demand_summary["Demand_Variation"] = (
    demand_summary["Demand_Variation"].fillna(0)
)

most_variable = demand_summary.loc[
    demand_summary["Demand_Variation"].idxmax()
]

print("\n3. DEMAND VARIATION INSIGHT")
print("Most Variable Product:", most_variable["Description"])
print(
    "Daily Demand Standard Deviation:",
    round(most_variable["Demand_Variation"], 2)
)

# 4. Inventory concentration insight
inventory_sorted = inventory.sort_values(
    "Total_Quantity", ascending=False
)

print("\n4. INVENTORY INSIGHT")
print("Top 5 Products by Units Sold:")

print(
    inventory_sorted[
        ["Description", "Total_Quantity", "Total_Transactions"]
    ].head(5).to_string(index=False)
)

# 5. Monthly revenue trend
daily_sales = pd.read_csv(
    DATA_DIR / "daily_sales.csv",
    parse_dates=["Date"]
)

daily_sales["Month"] = daily_sales["Date"].dt.to_period("M")

monthly_revenue = (
    daily_sales.groupby("Month")["Sales"].sum()
)

print("\n5. MONTHLY REVENUE TREND")

if len(monthly_revenue) >= 2:
    previous_month = monthly_revenue.iloc[-2]
    latest_month = monthly_revenue.iloc[-1]

    change = latest_month - previous_month

    if previous_month != 0:
        change_percent = (change / previous_month) * 100
    else:
        change_percent = np.nan

    print("Previous Month Revenue: £", round(previous_month, 2))
    print("Latest Month Revenue: £", round(latest_month, 2))

    if change > 0:
        print("Insight: Recorded monthly revenue increased.")
    elif change < 0:
        print("Insight: Recorded monthly revenue decreased.")
    else:
        print("Insight: Recorded monthly revenue remained unchanged.")

    if pd.notna(change_percent):
        print("Revenue Change (%):", round(change_percent, 2))

print("\nBusiness insights generated successfully!")
