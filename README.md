# AI Business Decision Intelligence

An AI-powered business analytics dashboard built using Python and Streamlit to help businesses analyze sales, forecast demand, monitor inventory, and evaluate business scenarios.

## Features

- **Sales Intelligence:** Analyze revenue, sales trends, and product performance.
- **Demand Forecasting:** Predict future sales using a Random Forest regression model.
- **Inventory Intelligence:** Estimate product demand and reorder points using lead time and safety stock inputs.
- **Business Simulator:** Explore hypothetical revenue scenarios by changing business assumptions.
- **Model Performance:** Compare AI predictions with baseline models using MAE and RMSE.
- **Data Upload:** Upload CSV and Excel datasets with supported column formats.
- **Data Validation:** Detect missing columns, invalid values, and duplicate records.
- **Export Reports:** Download analysis results in CSV and Excel formats.

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly
- Matplotlib
- Joblib

## Project Dataset

The project was initially developed and tested using the UCI Online Retail dataset.

It was also tested with a prepared version of the Brazilian Olist E-Commerce dataset.

Note: Olist order-item records represent order lines. They should not be interpreted as verified physical inventory quantities.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/aadiirana29o3-png/AI-Business-Decision-Intelligence.git
   ```

2. Open the project folder:

   ```bash
   cd AI-Business-Decision-Intelligence
   ```

3. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

4. Activate it on Windows:

   ```bash
   venv\Scripts\activate
   ```

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the Dashboard

```bash
streamlit run app/dashboard.py
```

## Dataset Format

The dashboard expects these columns:

| Column | Description |
|---|---|
| Date | Transaction or order date |
| Product | Product name or identifier |
| Quantity | Quantity sold |
| UnitPrice | Price per unit |

Supported CSV and Excel formats include `.csv`, `.xlsx`, and `.xls`.

## Important Notes

- Forecasts are estimates and depend on the quality and history of uploaded data.
- Reorder points are estimates based on average demand, lead time, and safety days.
- Business simulation results are hypothetical revenue scenarios, not guaranteed outcomes or profit predictions.
- Model performance metrics should be interpreted in the context of the dataset and evaluation method.

## Author

Developed as an AI/ML business intelligence project.
