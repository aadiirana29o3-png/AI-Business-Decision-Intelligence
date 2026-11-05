
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.business.model_performance import render_model_performance

st.set_page_config(
    page_title="AI Business Decision Intelligence",
    page_icon="📊",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/processed"
MODEL = ROOT / "src/models"

# Load data
@st.cache_data
def load_data():
    return {
        "daily": pd.read_csv(DATA / "daily_sales.csv", parse_dates=["Date"]),
        "inventory": pd.read_csv(DATA / "inventory_summary.csv"),
        "revenue": pd.read_csv(DATA / "profit_analysis.csv"),
        "products": pd.read_csv(DATA / "product_daily_sales.csv", parse_dates=["Date"]),
        "product_eval": pd.read_csv(DATA / "product_model_evaluation.csv")
    }


@st.cache_resource
def load_models():
    return {
        "sales": joblib.load(MODEL / "sales_forecast_model.pkl"),
        "product": joblib.load(MODEL / "product_demand_model.pkl")
    }


data = load_data()
models = load_models()

daily = data["daily"]
inventory = data["inventory"]
revenue = data["revenue"]
products = data["products"]

# Sidebar
st.sidebar.title("Business Intelligence")
st.sidebar.caption("AI-powered retail analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Business Overview",
        "Product Forecasting",
        "Inventory Intelligence",
        "Business Insights",
        "What-if Simulator",
        "Model Performance"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Dataset: UCI Online Retail")
st.sidebar.caption("Historical data: 2010–2011")


# Product forecasting helper
def predict_product(product_code, days):

    product_data = products[
        products["StockCode"].astype(str) == str(product_code)
    ].sort_values("Date").copy()

    if product_data.empty:
        return pd.DataFrame()

    history = product_data["Quantity"].astype(float).tolist()
    last_date = product_data["Date"].max()

    model = models["product"]
    predictions = []

    for i in range(days):

        next_date = last_date + pd.Timedelta(days=i + 1)

        previous_day = history[-1]
        last_7_days = np.mean(history[-7:])

        features = pd.DataFrame([{
            "Day": next_date.day,
            "Month": next_date.month,
            "DayOfWeek": next_date.dayofweek,
            "PreviousDaySales": previous_day,
            "SalesLast7Days": last_7_days
        }])

        prediction = max(0, float(model.predict(features)[0]))

        predictions.append({
            "Date": next_date,
            "PredictedDemand": round(prediction, 2)
        })

        history.append(prediction)

    return pd.DataFrame(predictions)


# 1. BUSINESS OVERVIEW
if page == "Business Overview":

    st.title("AI Business Decision Intelligence")
    st.write("A retail analytics platform for sales, demand and inventory decisions.")

    total_revenue = revenue["Total_Revenue"].sum()
    total_units = inventory["Total_Quantity"].sum()
    total_products = inventory["StockCode"].nunique()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Recorded Revenue", f"£{total_revenue:,.0f}")
    c2.metric("Total Units Sold", f"{total_units:,.0f}")
    c3.metric("Products Analysed", f"{total_products:,}")

    st.divider()

    st.subheader("Daily Sales Trend")

    chart_data = daily.set_index("Date")["Sales"]
    st.line_chart(chart_data)

    st.subheader("Top 10 Revenue Products")

    top_products = revenue.nlargest(10, "Total_Revenue")

    st.bar_chart(
        top_products.set_index("Description")["Total_Revenue"]
    )

    st.caption("Revenue is not profit. Product purchase costs are not available in this dataset.")


# 2. PRODUCT FORECASTING
elif page == "Product Forecasting":

    st.title("Product Demand Forecasting")
    st.write("Predict future product demand using the trained machine learning model.")

    product_options = products[
        ["StockCode", "Description"]
    ].drop_duplicates()

    product_options["Label"] = (
        product_options["Description"].astype(str)
        + " ("
        + product_options["StockCode"].astype(str)
        + ")"
    )

    selected_label = st.selectbox(
        "Select Product",
        product_options["Label"].tolist()
    )

    selected_code = product_options.loc[
        product_options["Label"] == selected_label,
        "StockCode"
    ].iloc[0]

    days = st.slider("Forecast Days", 1, 30, 7)

    if st.button("Predict Demand"):

        result = predict_product(selected_code, days)

        if not result.empty:

            st.subheader("Forecast Results")

            c1, c2 = st.columns(2)
            c1.metric(
                "Total Predicted Demand",
                f"{result['PredictedDemand'].sum():,.0f} units"
            )
            c2.metric(
                "Average Daily Demand",
                f"{result['PredictedDemand'].mean():,.1f} units"
            )

            st.line_chart(
                result.set_index("Date")["PredictedDemand"]
            )

            st.dataframe(result, use_container_width=True)

            st.download_button(
                "Download Forecast CSV",
                result.to_csv(index=False),
                file_name="product_demand_forecast.csv",
                mime="text/csv"
            )


# 3. INVENTORY INTELLIGENCE
elif page == "Inventory Intelligence":

    st.title("Inventory Intelligence")
    st.write("Estimate inventory requirements using predicted product demand.")

    product_options = products[
        ["StockCode", "Description"]
    ].drop_duplicates()

    product_options["Label"] = (
        product_options["Description"].astype(str)
        + " ("
        + product_options["StockCode"].astype(str)
        + ")"
    )

    selected_label = st.selectbox(
        "Select Product",
        product_options["Label"].tolist()
    )

    selected_code = product_options.loc[
        product_options["Label"] == selected_label,
        "StockCode"
    ].iloc[0]

    stock = st.number_input(
        "Current Available Stock (units)",
        min_value=0,
        value=100
    )

    lead_time = st.number_input(
        "Supplier Lead Time (days)",
        min_value=1,
        max_value=30,
        value=5
    )

    safety_days = st.number_input(
        "Additional Safety Days",
        min_value=0,
        max_value=30,
        value=2
    )

    if st.button("Analyse Inventory"):

        forecast_days = int(min(30, lead_time + safety_days))

        result = predict_product(selected_code, forecast_days)

        if not result.empty:

            expected_demand = result["PredictedDemand"].sum()
            reorder_qty = max(0, math.ceil(expected_demand - stock))

            st.subheader("Inventory Analysis")

            c1, c2, c3 = st.columns(3)

            c1.metric("Current Stock", f"{stock} units")
            c2.metric("Expected Demand", f"{expected_demand:.0f} units")
            c3.metric("Suggested Reorder", f"{reorder_qty} units")

            if stock < expected_demand:
                st.error("Potential Stock Shortage")
            else:
                st.success("Available stock covers the estimated demand.")

            st.caption(
                "This is a basic estimated stock gap. "
                "It does not account for existing purchase orders or forecast uncertainty."
            )

            st.dataframe(result, use_container_width=True)


# 4. BUSINESS INSIGHTS
elif page == "Business Insights":

    st.title("Business Insights")
    st.write("Explore important patterns in historical retail data.")

    highest_revenue = revenue.loc[
        revenue["Total_Revenue"].idxmax()
    ]

    highest_quantity = inventory.loc[
        inventory["Total_Quantity"].idxmax()
    ]

    product_variation = products.groupby(
        "Description"
    )["Quantity"].std()

    most_variable = product_variation.idxmax()

    c1, c2 = st.columns(2)

    c1.metric(
        "Highest Revenue Product",
        str(highest_revenue["Description"])
    )

    c2.metric(
        "Highest Quantity Product",
        str(highest_quantity["Description"])
    )

    st.metric("Most Variable Demand", str(most_variable))

    st.divider()
    st.subheader("Monthly Revenue Trend")

    monthly = daily.copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M")

    monthly_sales = monthly.groupby("Month")["Sales"].sum().reset_index()
    monthly_sales["Month"] = monthly_sales["Month"].astype(str)

    # Exclude incomplete final month (December 2011)
    monthly_sales = monthly_sales.iloc[:-1]

    st.bar_chart(
        monthly_sales.set_index("Month")["Sales"]
    )

    st.subheader("Top 5 Products by Quantity")

    top_five = inventory.nlargest(5, "Total_Quantity")

    st.dataframe(
        top_five[["Description", "Total_Quantity", "Total_Revenue"]],
        use_container_width=True
    )

    st.caption("Insights are descriptive analysis of historical data, not causal explanations.")


# 5. WHAT-IF SIMULATOR
elif page == "What-if Simulator":

    st.title("What-if Business Simulator")
    st.write("Explore how hypothetical demand and price changes affect revenue.")

    total_revenue = revenue["Total_Revenue"].sum()

    st.metric("Current Recorded Revenue", f"£{total_revenue:,.2f}")

    demand_change = st.slider(
        "Expected Demand Change (%)",
        -50, 100, 10
    )

    price_change = st.slider(
        "Selling Price Change (%)",
        -50, 100, 5
    )

    if st.button("Simulate Business Scenario"):

        estimated_revenue = (
            total_revenue
            * (1 + demand_change / 100)
            * (1 + price_change / 100)
        )

        difference = estimated_revenue - total_revenue

        c1, c2 = st.columns(2)

        c1.metric(
            "Estimated Revenue",
            f"£{estimated_revenue:,.2f}"
        )

        c2.metric(
            "Revenue Difference",
            f"£{difference:,.2f}",
            delta=f"{difference:,.2f}"
        )

        st.caption(
            "Illustrative scenario only. It assumes demand and price changes "
            "are independent and does not model customer price sensitivity."
        )


# 6. MODEL PERFORMANCE
elif page == "Model Performance":

    render_model_performance()

