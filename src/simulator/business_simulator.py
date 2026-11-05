
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Load revenue data
file_path = BASE_DIR / "data" / "processed" / "profit_analysis.csv"
df = pd.read_csv(file_path)

current_revenue = df["Total_Revenue"].sum()

print("\n===== AI BUSINESS DECISION INTELLIGENCE =====")
print("Current Total Revenue: £", round(current_revenue, 2))

# Take business decisions
demand_change = float(input("\nExpected demand change (%): "))
price_change = float(input("Expected selling price change (%): "))

# Validate inputs
if demand_change < -100 or price_change < -100:
    print("\nError: Percentage change cannot be below -100%.")
else:
    demand_factor = 1 + demand_change / 100
    price_factor = 1 + price_change / 100

    # Calculate revenue after demand change
    revenue_after_demand = current_revenue * demand_factor

    # Calculate revenue after price change
    estimated_revenue = revenue_after_demand * price_factor

    # Calculate differences
    demand_impact = revenue_after_demand - current_revenue
    price_impact = estimated_revenue - revenue_after_demand
    total_difference = estimated_revenue - current_revenue

    print("\n===== BUSINESS DECISION ANALYSIS =====")
    print("Current Revenue: £", round(current_revenue, 2))
    print("Revenue after Demand Change: £", round(revenue_after_demand, 2))
    print("Revenue after Price Change: £", round(estimated_revenue, 2))

    print("\n===== IMPACT ANALYSIS =====")
    print("Demand Impact: £", round(demand_impact, 2))
    print("Price Impact: £", round(price_impact, 2))
    print("Total Revenue Change: £", round(total_difference, 2))

    if total_difference > 0:
        print("\nDecision: Estimated revenue increased.")
    elif total_difference < 0:
        print("\nDecision: Estimated revenue decreased.")
    else:
        print("\nDecision: No estimated revenue change.")
