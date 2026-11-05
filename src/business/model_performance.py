
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]

def render_model_performance():

    st.title("Model Performance & Accuracy")
    st.write("Evaluate AI predictions against actual sales and a simple baseline model.")

    sales_path = ROOT / "data/processed/forecast_results.csv"
    product_path = ROOT / "data/processed/product_model_evaluation.csv"

    sales_df = pd.read_csv(sales_path, parse_dates=["Date"])
    product_df = pd.read_csv(product_path)

    # Overall sales model metrics
    actual = sales_df["ActualSales"]
    predicted = sales_df["PredictedSales"]

    sales_mae = mean_absolute_error(actual, predicted)
    sales_rmse = np.sqrt(mean_squared_error(actual, predicted))

    st.subheader("Overall Sales Forecast Model")

    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"£{sales_mae:,.2f}")
    col2.metric("Root Mean Squared Error (RMSE)", f"£{sales_rmse:,.2f}")

    st.caption(
        "MAE shows the average absolute prediction error. "
        "RMSE gives larger errors more weight."
    )

    st.subheader("Actual vs Predicted Sales")

    chart_df = sales_df.set_index("Date")[["ActualSales", "PredictedSales"]]
    st.line_chart(chart_df)

    # Product model evaluation
    st.divider()
    st.subheader("Product Demand Model")

    better = int((product_df["Improvement_Percent"] > 0).sum())
    worse = int((product_df["Improvement_Percent"] < 0).sum())
    total = len(product_df)

    avg_model_mae = product_df["Model_MAE"].mean()
    avg_baseline_mae = product_df["Baseline_MAE"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Products Evaluated", total)
    col2.metric("AI Better Than Baseline", better)
    col3.metric("AI Worse Than Baseline", worse)

    col1, col2 = st.columns(2)
    col1.metric("Average AI Model MAE", f"{avg_model_mae:.2f}")
    col2.metric("Average Baseline MAE", f"{avg_baseline_mae:.2f}")

    st.caption(
        "Product MAE values are averaged across the evaluated products. "
        "Lower MAE means lower average prediction error."
    )

    st.subheader("Product-wise Model Improvement")

    comparison = product_df[
        ["Description", "Improvement_Percent"]
    ].sort_values("Improvement_Percent")

    st.bar_chart(
        comparison.set_index("Description")["Improvement_Percent"]
    )

    st.subheader("Detailed Evaluation")

    st.dataframe(
        product_df.sort_values("Improvement_Percent", ascending=False),
        use_container_width=True
    )
