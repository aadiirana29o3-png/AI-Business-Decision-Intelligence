import streamlit as st
import pandas as pd
import numpy as np
import io
import hashlib
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Business Decision Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROFESSIONAL BLUE THEME
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: #081426;
        color: #edf4ff;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    [data-testid="stAppViewContainer"] .main p,
    [data-testid="stAppViewContainer"] .main label,
    [data-testid="stAppViewContainer"] .main h1,
    [data-testid="stAppViewContainer"] .main h2,
    [data-testid="stAppViewContainer"] .main h3,
    [data-testid="stAppViewContainer"] .main h4 {
        color: #edf4ff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0B192B;
        border-right: 1px solid #263b5c;
    }

    [data-testid="stSidebar"] * {
        color: #e5efff;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #aebfda;
    }

    /* Hero */
    .hero {
        padding: 27px 30px;
        border-radius: 16px;
        background: linear-gradient(115deg, #102442, #238B8D);
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 6px 22px rgba(0, 0, 0, 0.2);
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 32px;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #dceaff !important;
        margin: 0;
        font-size: 15px;
    }

    /* KPI cards */
    .metric-card {
        background: linear-gradient(135deg, #0D1B2E, #142A43);
        border: 1px solid #29445F;
        border-radius: 13px;
        padding: 19px;
        min-height: 105px;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.16);
    }

    .metric-label {
        color: #c2d7f5 !important;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #ffffff !important;
        font-size: 25px;
        line-height: 1.3;
        font-weight: 750;
        overflow-wrap: anywhere;
    }

    /* Streamlit metrics */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0D1B2E, #142A43);
        border: 1px solid #29445F;
        border-radius: 12px;
        padding: 14px 17px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p {
        color: #c2d7f5 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: #ffffff !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #80e6b0 !important;
    }

    /* Inputs */
    .stSelectbox label,
    .stMultiSelect label,
    .stSlider label,
    .stNumberInput label,
    .stDateInput label,
    .stTextInput label,
    .stFileUploader label {
        color: #e7f0ff !important;
        font-weight: 600;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: #0D1B2E !important;
        border: 1px solid #29445F !important;
        border-radius: 9px !important;
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        color: #ffffff !important;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="menu"] {
        background: #0D1B2E !important;
    }

    div[data-baseweb="menu"] * {
        color: #ffffff !important;
    }

    /* Buttons */
    .stButton button,
    .stDownloadButton button {
        background: linear-gradient(100deg, #2459bd, #2BA3A0);
        color: white !important;
        border: 1px solid #48A9A5;
        border-radius: 9px;
        font-weight: 650;
    }

    .stButton button:hover,
    .stDownloadButton button:hover {
        background: #35B5AD;
        border-color: #8BCBC5;
        color: white !important;
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #304c70;
        border-radius: 10px;
        overflow: hidden;
    }

    .small-note {
        color: #aebfda;
        font-size: 13px;
    }

    /* Sidebar dropdown */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #102442 !important;
        border: 1px solid #29445F !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. COMMON DISPLAY FUNCTIONS
# ============================================================

def show_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_metric(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def fmt(value):
    return f"{value:,.2f}"


def style_chart(fig, title=None):
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=20, r=20, t=55, b=25),
        font=dict(family="Arial", size=13),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_color="#14213d"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        showspikes=False,
        spikethickness=0
    )

    fig.update_yaxes(
        showspikes=False,
        spikethickness=0
    )

    return fig


def show_chart(fig):
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d"
            ]
        }
    )


# ============================================================
# 4. COLUMN NORMALIZATION AND CLEANING
# ============================================================

COLUMN_ALIASES = {
    "date": [
        "date", "invoicedate", "orderdate",
        "transactiondate", "datetime"
    ],
    "product": [
        "product", "description", "productname",
        "stockcode", "item", "itemname"
    ],
    "quantity": [
        "quantity", "qty", "unitssold",
        "units", "salesquantity"
    ],
    "unitprice": [
        "unitprice", "price", "sellingprice",
        "salesprice", "rate"
    ]
}


def normalize_columns(df):
    df = df.copy()

    normalized = {
        str(col).strip().lower().replace(" ", "").replace("_", ""): col
        for col in df.columns
    }

    rename_map = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = alias.replace(" ", "").replace("_", "")

            if key in normalized:
                rename_map[normalized[key]] = standard_name.title()
                break

    return df.rename(columns=rename_map)


def clean_data(raw_df):
    df = normalize_columns(raw_df)

    required = ["Date", "Product", "Quantity", "Unitprice"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Required columns missing: " + ", ".join(missing)
        )

    original_rows = len(df)

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    missing_product_count = int(df["Product"].isna().sum())

    df = df.dropna(
        subset=["Date", "Product", "Quantity", "Unitprice"]
    ).copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    invalid_date_count = int(df["Date"].isna().sum())
    df = df.dropna(subset=["Date"]).copy()

    
    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["Unitprice"] = pd.to_numeric(
        df["Unitprice"],
        errors="coerce"
    )

# Count invalid values BEFORE removing rows
    invalid_numeric_count = int(
        (df["Quantity"].isna() | df["Unitprice"].isna()).sum()
    )

    invalid_quantity_count = int(
        (df["Quantity"].isna() | (df["Quantity"] <= 0)).sum()
    )

    invalid_price_count = int(
        (df["Unitprice"].isna() | (df["Unitprice"] <= 0)).sum()
    )

# Remove missing numeric values
    df = df.dropna(
        subset=["Quantity", "Unitprice"]
    ).copy()


    df = df[
        (df["Quantity"] > 0) &
        (df["Unitprice"] > 0)
    ].copy()

    df["Product"] = df["Product"].astype(str).str.strip()
    df = df[df["Product"] != ""].copy()

    df["Revenue"] = df["Quantity"] * df["Unitprice"]

    df = df.sort_values("Date").reset_index(drop=True)

    report = {
        "Original Rows": original_rows,
        "Duplicate Rows": duplicate_count,
        "Missing Product Rows": missing_product_count,
        "Invalid Dates": invalid_date_count,
        "Invalid Numeric Rows": invalid_numeric_count,
        "Invalid Quantity Rows": invalid_quantity_count,
        "Invalid Price Rows": invalid_price_count,
        "Cleaned Rows": len(df)
    }

    return df, report


# ============================================================
# 5. FAST FILE READING
# ============================================================

@st.cache_data(show_spinner=False, max_entries=3)
def load_and_clean_file(file_bytes, file_name):
    extension = file_name.lower()

    if extension.endswith(".csv"):
        raw_df = pd.read_csv(
            io.BytesIO(file_bytes),
            low_memory=False
        )
    else:
        try:
            raw_df = pd.read_excel(
                io.BytesIO(file_bytes),
                engine="calamine"
            )
        except (ImportError, ValueError):
            raw_df = pd.read_excel(
                io.BytesIO(file_bytes)
            )

    cleaned_df, report = clean_data(raw_df)

    return cleaned_df, report


# ============================================================
# 6. DATA ANALYSIS FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False, max_entries=5)
def get_daily_sales(df):
    daily = (
        df.groupby(df["Date"].dt.floor("D"))["Revenue"]
        .sum()
        .sort_index()
    )

    full_dates = pd.date_range(
        daily.index.min(),
        daily.index.max(),
        freq="D"
    )

    daily = daily.reindex(
        full_dates,
        fill_value=0
    )

    daily.index.name = "Date"

    return daily


@st.cache_data(show_spinner=False, max_entries=5)
def get_monthly_sales(df):
    monthly = (
        df.set_index("Date")["Revenue"]
        .resample("MS")
        .sum()
        .reset_index()
    )

    monthly["Month"] = monthly["Date"].dt.strftime("%b %Y")

    return monthly


@st.cache_data(show_spinner=False, max_entries=5)
def get_product_revenue(df):
    return (
        df.groupby("Product")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


@st.cache_data(show_spinner=False, max_entries=5)
def get_daily_quantity(df):
    result = (
        df.groupby(df["Date"].dt.floor("D"))["Quantity"]
        .sum()
        .reset_index()
    )

    result.columns = ["Date", "Quantity"]
    return result


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

def create_features(series):
    data = pd.DataFrame({
        "Date": series.index,
        "Sales": series.values
    })

    data["Day"] = data["Date"].dt.day
    data["Month"] = data["Date"].dt.month
    data["DayOfWeek"] = data["Date"].dt.dayofweek

    data["PreviousDaySales"] = data["Sales"].shift(1)

    data["SalesLast7Days"] = (
        data["Sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    data = data.dropna().reset_index(drop=True)

    features = [
        "Day",
        "Month",
        "DayOfWeek",
        "PreviousDaySales",
        "SalesLast7Days"
    ]

    return data, features


# ============================================================
# 8. MACHINE LEARNING MODEL
# ============================================================

@st.cache_data(show_spinner=False, max_entries=10)
def train_sales_model(series):
    if len(series) < 35:
        return None

    data, features = create_features(series)

    if len(data) < 25:
        return None

    split = int(len(data) * 0.8)

    train = data.iloc[:split]
    test = data.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[features],
        train["Sales"]
    )

    predictions = model.predict(test[features])

    mae = mean_absolute_error(
        test["Sales"],
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["Sales"],
            predictions
        )
    )

    baseline = test["PreviousDaySales"].values

    baseline_mae = mean_absolute_error(
        test["Sales"],
        baseline
    )

    evaluation = pd.DataFrame({
        "Date": test["Date"].values,
        "Actual Sales": test["Sales"].values,
        "Predicted Sales": predictions,
        "Baseline Prediction": baseline
    })

    model.fit(
        data[features],
        data["Sales"]
    )

    return (
        model,
        features,
        mae,
        rmse,
        baseline_mae,
        evaluation
    )


def forecast_future(series, model, features, days):
    history = list(series.values)
    last_date = series.index.max()

    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=days,
        freq="D"
    )

    predictions = []

    for future_date in future_dates:

        previous_day = history[-1]
        last_7_days = float(np.mean(history[-7:]))

        row = pd.DataFrame([{
            "Day": future_date.day,
            "Month": future_date.month,
            "DayOfWeek": future_date.dayofweek,
            "PreviousDaySales": previous_day,
            "SalesLast7Days": last_7_days
        }])

        prediction = float(
            model.predict(row[features])[0]
        )

        prediction = max(0, prediction)

        predictions.append(prediction)
        history.append(prediction)

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted Sales": predictions
    })


# ============================================================
# 9. PRODUCT DEMAND FORECASTING
# ============================================================

@st.cache_data(show_spinner=False, max_entries=20)
def get_product_daily_series(df, product):
    product_df = df[df["Product"] == product]

    daily = (
        product_df.groupby(
            product_df["Date"].dt.floor("D")
        )["Quantity"]
        .sum()
        .sort_index()
    )

    dates = pd.date_range(
        daily.index.min(),
        daily.index.max(),
        freq="D"
    )

    return daily.reindex(
        dates,
        fill_value=0
    )


def forecast_product_demand(df, product, days):
    series = get_product_daily_series(
        df,
        product
    )

    if len(series) < 35:

        avg_demand = float(
            series.tail(30).mean()
        )

        last_date = series.index.max()

        dates = pd.date_range(
            last_date + pd.Timedelta(days=1),
            periods=days,
            freq="D"
        )

        future = pd.DataFrame({
            "Date": dates,
            "Predicted Demand": [
                max(0, avg_demand)
            ] * days
        })

        return (
            future,
            "Recent daily average (limited history)",
            None
        )

    trained = train_sales_model(series)

    if trained is None:
        return None, "Insufficient data", None

    model, features, mae, rmse, baseline_mae, evaluation = trained

    future = forecast_future(
        series,
        model,
        features,
        days
    )

    future = future.rename(
        columns={
            "Predicted Sales": "Predicted Demand"
        }
    )

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "Baseline MAE": baseline_mae
    }

    return future, "Random Forest Regression", metrics


# ============================================================
# 10. FAST INVENTORY INTELLIGENCE
# ============================================================

@st.cache_data(show_spinner=False, max_entries=5)
def calculate_inventory(df):
    latest_date = df["Date"].max().floor("D")
    start_date = latest_date - pd.Timedelta(days=29)

    recent = df[
        (df["Date"] >= start_date) &
        (df["Date"] < latest_date + pd.Timedelta(days=1))
    ]

    summary = (
        recent.groupby("Product", sort=False)
        .agg(
            Total_Quantity=("Quantity", "sum")
        )
        .reset_index()
    )

    summary["Average Daily Demand"] = (
        summary["Total_Quantity"] / 30
    )

    summary["Lead Time Days"] = 7
    summary["Safety Days"] = 5

    summary["Estimated Reorder Point"] = (
        summary["Average Daily Demand"] * 12
    ).round().astype(int)

    summary = summary.sort_values(
        "Average Daily Demand",
        ascending=False
    ).reset_index(drop=True)

    return summary


# ============================================================
# 11. EXCEL SUMMARY REPORT GENERATOR
# ============================================================

def create_summary_excel(report_data):
    output = io.BytesIO()

    try:
        with pd.ExcelWriter(
            output,
            engine="xlsxwriter",
            datetime_format="dd-mmm-yyyy"
                ) as writer:

            for sheet_name, data in report_data.items():

                data.to_excel(
                    writer,
                    sheet_name=sheet_name[:31],
                    index=False
                )

                worksheet = writer.sheets[sheet_name[:31]]

                worksheet.freeze_panes(1, 0)

                worksheet.autofilter(
                    0,
                    0,
                    len(data),
                    max(0, len(data.columns) - 1)
                )

                for index, column in enumerate(data.columns):

                    worksheet.set_column(
                        index,
                        index,
                        min(max(len(str(column)) + 4, 14), 30)
                    )

    except ImportError:

        output = io.BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            for sheet_name, data in report_data.items():

                data.to_excel(
                    writer,
                    sheet_name=sheet_name[:31],
                    index=False
                )

    output.seek(0)

    return output.getvalue()


# ============================================================
# 12. SESSION STATE
# ============================================================

if "business_df" not in st.session_state:
    st.session_state.business_df = None

if "cleaning_report" not in st.session_state:
    st.session_state.cleaning_report = None

if "uploaded_file_hash" not in st.session_state:
    st.session_state.uploaded_file_hash = None

if "sales_forecast" not in st.session_state:
    st.session_state.sales_forecast = None

if "product_future" not in st.session_state:
    st.session_state.product_future = None

if "sales_evaluation" not in st.session_state:
    st.session_state.sales_evaluation = None

if "sales_model_metrics" not in st.session_state:
    st.session_state.sales_model_metrics = None


# ============================================================
# 13. SIDEBAR NAVIGATION - DROPDOWN
# ============================================================

st.sidebar.markdown("""
<div style="padding: 12px 4px 18px 4px;">
    <div style="font-size: 27px; font-weight: 800; color: white;">
        📊 AI Business
    </div>
    <div style="font-size: 21px; font-weight: 700; color: white;">
        Intelligence
    </div>
    <div style="font-size: 13px; color: #aebbd0; margin-top: 8px;">
        Business analytics platform
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

pages = [
    "Data Upload",
    "Business Overview",
    "Sales Analysis",
    "Demand Forecasting",
    "Product Forecasting",
    "Model Performance",
    "Inventory Intelligence",
    "Business Insights",
    "What-if Simulator",
    "Download Reports"
]

selected_page = st.sidebar.selectbox(
    "SELECT WORKSPACE",
    pages,
    index=0,
    key="page_navigation"
)

st.sidebar.divider()

if st.session_state.business_df is not None:
    st.sidebar.success("Dataset loaded")
    st.sidebar.caption(
        f"{len(st.session_state.business_df):,} cleaned records"
    )
else:
    st.sidebar.info("No dataset loaded")

st.sidebar.markdown("""
<div style="padding-top: 15px; color: #8795aa; font-size: 12px;">
    MACHINE LEARNING<br>
    BUSINESS ANALYTICS<br>
    DEMAND FORECASTING
</div>
""", unsafe_allow_html=True)


# ============================================================
# 14. CURRENT DATA
# ============================================================

df = st.session_state.business_df


# ============================================================
# 15. DATA UPLOAD PAGE
# ============================================================

if selected_page == "Data Upload":

    show_header(
        "Data Upload & Cleaning",
        "Upload your retail dataset and prepare it for business analysis."
    )

    st.markdown("### Upload your dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if file_hash != st.session_state.uploaded_file_hash:

            try:
                with st.spinner(
                    "Reading and cleaning your dataset..."
                ):

                    cleaned_df, report = load_and_clean_file(
                        file_bytes,
                        uploaded_file.name
                    )

                st.session_state.business_df = cleaned_df
                st.session_state.cleaning_report = report
                st.session_state.uploaded_file_hash = file_hash

                st.session_state.sales_forecast = None
                st.session_state.product_future = None
                st.session_state.sales_evaluation = None
                st.session_state.sales_model_metrics = None

                st.success(
                    "Dataset uploaded and cleaned successfully."
                )

                st.rerun()

            except Exception as e:
                st.error(f"Could not load this file: {e}")

    df = st.session_state.business_df

    if df is not None:

        st.markdown("### Cleaning Summary")

        report_df = pd.DataFrame(
            list(st.session_state.cleaning_report.items()),
            columns=["Cleaning Step", "Rows"]
        )

        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Cleaned Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True,
            hide_index=True
        )

        a, b, c = st.columns(3)

        a.metric("Cleaned Records", f"{len(df):,}")
        b.metric("Unique Products", f"{df['Product'].nunique():,}")
        c.metric("Total Revenue", fmt(df["Revenue"].sum()))

        st.caption(
            "Required columns: Date, Product, Quantity and UnitPrice. "
            "Common column aliases are supported."
        )


# ============================================================
# 16. DATA REQUIRED CHECK
# ============================================================

elif df is None:

    show_header(
        "No Dataset Available",
        "Upload a compatible CSV or Excel dataset to start business analysis."
    )

    st.warning(
        "Open Data Upload from the sidebar and upload your dataset first."
    )


# ============================================================
# 17. BUSINESS OVERVIEW
# ============================================================

elif selected_page == "Business Overview":

    show_header(
        "Business Overview",
        "A complete snapshot of your business performance."
    )

    daily_sales = get_daily_sales(df)

    total_revenue = float(df["Revenue"].sum())
    total_quantity = int(df["Quantity"].sum())
    total_products = int(df["Product"].nunique())

    a, b = st.columns(2)

    with a:
        show_metric("Total Revenue", fmt(total_revenue))

    with b:
        show_metric("Total Quantity Sold", f"{total_quantity:,}")

    c1, c2, c3 = st.columns(3)

    c1.metric("Unique Products", f"{total_products:,}")
    c2.metric("Total Transactions / Rows", f"{len(df):,}")
    c3.metric("Average Daily Revenue", fmt(daily_sales.mean()))

    st.caption(
        "Revenue = Quantity × UnitPrice. Currency depends on the uploaded "
        "dataset. UCI Online Retail values are in GBP."
    )

    st.markdown("### Daily Revenue Trend")

    daily_display = daily_sales.reset_index()

    fig = px.line(
        daily_display,
        x="Date",
        y="Revenue",
        labels={"Date": "Date", "Revenue": "Revenue"}
    )

    fig.update_traces(
        line=dict(width=2),
        hovertemplate=(
            "Date: %{x|%d %b %Y}<br>"
            "Revenue: %{y:,.2f}"
            "<extra></extra>"
        )
    )

    show_chart(style_chart(fig, "Daily Revenue"))

    st.markdown("### Monthly Revenue")

    monthly = get_monthly_sales(df)

    fig = px.bar(
        monthly,
        x="Month",
        y="Revenue",
        labels={"Month": "Month", "Revenue": "Revenue"}
    )

    fig.update_traces(
        hovertemplate=(
            "Month: %{x}<br>"
            "Revenue: %{y:,.2f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(xaxis_tickangle=-35)

    show_chart(style_chart(fig, "Monthly Revenue"))


# ============================================================
# 18. SALES ANALYSIS
# ============================================================

elif selected_page == "Sales Analysis":

    show_header(
        "Sales Analysis",
        "Explore revenue trends, product performance and sales patterns."
    )

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=df["Date"].min().date(),
            min_value=df["Date"].min().date(),
            max_value=df["Date"].max().date()
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=df["Date"].max().date(),
            min_value=df["Date"].min().date(),
            max_value=df["Date"].max().date()
        )

    if start_date > end_date:

        st.error("Start Date cannot be after End Date.")

    else:

        filtered = df[
            (df["Date"].dt.date >= start_date) &
            (df["Date"].dt.date <= end_date)
        ]

        if filtered.empty:

            st.warning("No records found for this date range.")

        else:

            revenue = float(filtered["Revenue"].sum())
            quantity = float(filtered["Quantity"].sum())
            unique_products = int(filtered["Product"].nunique())

            c1, c2, c3 = st.columns(3)

            c1.metric("Revenue", fmt(revenue))
            c2.metric("Quantity Sold", f"{quantity:,.0f}")
            c3.metric("Unique Products", f"{unique_products:,}")

            st.caption(
                "Unique Products means distinct product names sold in this "
                "date range. It is not the number of units sold."
            )

            st.markdown("### Top 10 Products by Revenue")

            top_products = (
                filtered.groupby("Product")["Revenue"]
                .sum()
                .nlargest(10)
                .reset_index()
                .sort_values("Revenue")
            )

            fig = px.bar(
                top_products,
                x="Revenue",
                y="Product",
                orientation="h"
            )

            fig.update_traces(
                hovertemplate=(
                    "Product: %{y}<br>"
                    "Revenue: %{x:,.2f}"
                    "<extra></extra>"
                )
            )

            show_chart(style_chart(fig, "Top Products by Revenue"))

            st.markdown("### Daily Sales Quantity")

            daily_quantity = (
                filtered.groupby(
                    filtered["Date"].dt.floor("D")
                )["Quantity"]
                .sum()
                .reset_index()
            )

            daily_quantity.columns = ["Date", "Quantity"]

            fig = px.area(
                daily_quantity,
                x="Date",
                y="Quantity"
            )

            fig.update_traces(
                hovertemplate=(
                    "Date: %{x|%d %b %Y}<br>"
                    "Units: %{y:,.0f}"
                    "<extra></extra>"
                )
            )

            show_chart(style_chart(fig, "Daily Quantity Sold"))


# ============================================================
# 19. DEMAND FORECASTING
# ============================================================

elif selected_page == "Demand Forecasting":

    show_header(
        "Sales Demand Forecasting",
        "Predict future daily revenue using a machine learning model."
    )

    daily_sales = get_daily_sales(df)

    st.write(
        f"Historical data: **{daily_sales.index.min().date()}** "
        f"to **{daily_sales.index.max().date()}**."
    )

    forecast_days = st.number_input(
        "How many future days should be forecast?",
        min_value=1,
        max_value=365,
        value=30,
        step=1
    )

    if st.button("Generate Sales Forecast", type="primary"):

        with st.spinner("Training model and generating forecast..."):

            trained = train_sales_model(daily_sales)

            if trained is None:

                st.error(
                    "At least 35 calendar days of historical data "
                    "are required for this model."
                )

            else:

                model, features, mae, rmse, baseline_mae, evaluation = trained

                future = forecast_future(
                    daily_sales,
                    model,
                    features,
                    int(forecast_days)
                )

                st.session_state.sales_forecast = future
                st.session_state.sales_evaluation = evaluation

                st.session_state.sales_model_metrics = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "Baseline MAE": baseline_mae
                }

                st.success("Forecast generated successfully.")

    future = st.session_state.sales_forecast

    if future is not None:

        st.markdown("### Forecast Results")

        c1, c2 = st.columns(2)

        c1.metric("Forecast Period", f"{len(future)} Days")

        c2.metric(
            "Forecast Revenue",
            fmt(future["Predicted Sales"].sum())
        )

        fig = go.Figure()

        history = daily_sales.tail(90).reset_index()

        fig.add_trace(
            go.Scatter(
                x=history["Date"],
                y=history["Revenue"],
                mode="lines",
                name="Historical Revenue"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=future["Date"],
                y=future["Predicted Sales"],
                mode="lines+markers",
                name="Forecast Revenue",
                line=dict(dash="dash", width=3)
            )
        )

        fig.add_vline(
            x=daily_sales.index.max(),
            line_dash="dot",
            annotation_text="Forecast starts"
        )

        fig.update_traces(
            hovertemplate=(
                "Date: %{x|%d %b %Y}<br>"
                "Revenue: %{y:,.2f}"
                "<extra></extra>"
            )
        )

        show_chart(style_chart(fig, "Historical vs Forecast Revenue"))

        st.dataframe(
            future,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Forecast CSV",
            data=future.to_csv(index=False).encode("utf-8"),
            file_name="sales_forecast.csv",
            mime="text/csv"
        )


# ============================================================
# 20. PRODUCT FORECASTING
# ============================================================

elif selected_page == "Product Forecasting":

    show_header(
        "Product Demand Forecasting",
        "Estimate future demand for an individual product."
    )

    products = sorted(
        df["Product"].dropna().unique().tolist()
    )

    selected_product = st.selectbox(
        "Select a Product",
        products
    )

    forecast_days = st.number_input(
        "Forecast duration (days)",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        key="product_forecast_days"
    )

    if st.button("Generate Product Forecast", type="primary"):

        with st.spinner("Analysing product demand..."):

            future, method, metrics = forecast_product_demand(
                df,
                selected_product,
                int(forecast_days)
            )

            if future is None:

                st.error(
                    "Not enough historical data to generate this forecast."
                )

            else:

                st.session_state.product_future = future
                st.session_state.product_method = method
                st.session_state.product_metrics = metrics
                st.session_state.product_selected = selected_product

                st.success("Product forecast generated.")

    if (
        st.session_state.product_future is not None
        and st.session_state.product_selected == selected_product
    ):

        future = st.session_state.product_future
        method = st.session_state.product_method
        metrics = st.session_state.product_metrics

        st.info(f"Forecast method: **{method}**")

        c1, c2 = st.columns(2)

        c1.metric("Forecast Duration", f"{len(future)} Days")

        c2.metric(
            "Expected Demand",
            f"{future['Predicted Demand'].sum():,.0f} Units"
        )

        history = get_product_daily_series(
            df,
            selected_product
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=history.tail(90).index,
                y=history.tail(90).values,
                mode="lines",
                name="Historical Demand"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=future["Date"],
                y=future["Predicted Demand"],
                mode="lines+markers",
                name="Predicted Demand",
                line=dict(dash="dash", width=3)
            )
        )

        fig.add_vline(
            x=history.index.max(),
            line_dash="dot",
            annotation_text="Forecast starts"
        )

        fig.update_traces(
            hovertemplate=(
                "Date: %{x|%d %b %Y}<br>"
                "Units: %{y:,.2f}"
                "<extra></extra>"
            )
        )

        show_chart(
            style_chart(fig, f"Demand Forecast: {selected_product}")
        )

        if metrics is not None:

            st.markdown("### Model Evaluation")

            a, b, c = st.columns(3)

            a.metric("MAE", fmt(metrics["MAE"]))
            b.metric("RMSE", fmt(metrics["RMSE"]))
            c.metric("Baseline MAE", fmt(metrics["Baseline MAE"]))

            st.caption(
                "Historical test metrics do not change when the future "
                "forecast duration changes."
            )

        st.dataframe(
            future,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Product Forecast",
            data=future.to_csv(index=False).encode("utf-8"),
            file_name="product_demand_forecast.csv",
            mime="text/csv"
        )


# ============================================================
# 21. MODEL PERFORMANCE
# ============================================================

elif selected_page == "Model Performance":

    show_header(
        "Model Performance",
        "Evaluate the sales forecasting model against a simple baseline."
    )

    daily_sales = get_daily_sales(df)

    with st.spinner("Evaluating model..."):
        trained = train_sales_model(daily_sales)

    if trained is None:

        st.warning(
            "At least 35 calendar days of historical data are required."
        )

    else:

        model, features, mae, rmse, baseline_mae, evaluation = trained

        improvement = (
            (baseline_mae - mae) / baseline_mae * 100
            if baseline_mae != 0 else 0
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Model MAE", fmt(mae))
        c2.metric("Model RMSE", fmt(rmse))
        c3.metric("Baseline MAE", fmt(baseline_mae))

        st.metric(
            "MAE Improvement vs Baseline",
            f"{improvement:+.2f}%"
        )

        st.markdown("### Actual vs Predicted Revenue")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=evaluation["Date"],
                y=evaluation["Actual Sales"],
                mode="lines",
                name="Actual Revenue"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=evaluation["Date"],
                y=evaluation["Predicted Sales"],
                mode="lines",
                name="Predicted Revenue"
            )
        )

        show_chart(style_chart(fig, "Actual vs Predicted Revenue"))

        st.markdown("### Model vs Baseline Error")

        comparison = pd.DataFrame({
            "Model": [
                "Random Forest",
                "Previous Day Baseline"
            ],
            "MAE": [
                mae,
                baseline_mae
            ]
        })

        fig = px.bar(
            comparison,
            x="Model",
            y="MAE",
            text_auto=".2f"
        )

        fig.update_traces(
            hovertemplate=(
                "Model: %{x}<br>"
                "MAE: %{y:,.2f}"
                "<extra></extra>"
            )
        )

        show_chart(style_chart(fig, "MAE Comparison"))

        st.caption(
            "Lower MAE means lower average absolute prediction error. "
            "These results apply only to the currently uploaded dataset "
            "and its chronological test split."
        )

        st.download_button(
            "Download Model Evaluation",
            data=evaluation.to_csv(index=False).encode("utf-8"),
            file_name="model_evaluation.csv",
            mime="text/csv"
        )


# ============================================================
# 22. INVENTORY INTELLIGENCE
# ============================================================

elif selected_page == "Inventory Intelligence":

    show_header(
        "Inventory Intelligence",
        "Explore overall inventory demand and inspect individual products."
    )

    with st.spinner("Calculating inventory intelligence..."):
        inventory = calculate_inventory(df)

    if inventory.empty:

        st.warning("No inventory information available.")

    else:
        
        # --------------------------------------------------------
        # OVERALL INVENTORY OVERVIEW
        # --------------------------------------------------------

        c1, c2, c3 = st.columns(3)

        c1.metric("Products Analysed", f"{len(inventory):,}")

        c2.metric(
            "Highest Daily Demand",
            f"{inventory['Average Daily Demand'].max():,.2f} units"
        )

        c3.metric(
            "Average Daily Demand",
            f"{inventory['Average Daily Demand'].mean():,.2f} units"
        )

        st.markdown("### Top 15 Products by Daily Demand")

        top_inventory = (
            inventory.head(15)
            .sort_values("Average Daily Demand")
        )

        fig = px.bar(
            top_inventory,
            x="Average Daily Demand",
            y="Product",
            orientation="h",
            labels={
                "Average Daily Demand": "Average Daily Demand (units/day)",
                "Product": "Product"
            }
        )

        fig.update_traces(
            hovertemplate=(
                "Product: %{y}<br>"
                "Average Daily Demand: %{x:.2f} units/day"
                "<extra></extra>"
            )
        )

        show_chart(style_chart(fig, "Highest Demand Products"))

        # --------------------------------------------------------
        # MANUAL PRODUCT-WISE INVENTORY ANALYSIS
        # --------------------------------------------------------

        st.markdown("---")
        st.markdown("### Individual Product Analysis")
        st.write(
            "Select any product to inspect its recent demand and "
            "estimated reorder requirement."
        )
                # --------------------------------------------------------
        # MANUAL INVENTORY PLANNING SETTINGS
        # --------------------------------------------------------

        st.markdown("#### Inventory Planning Settings")

        setting_col1, setting_col2 = st.columns(2)

        with setting_col1:
            lead_time_days = st.number_input(
                "Supplier Lead Time (days)",
                min_value=1,
                max_value=365,
                value=7,
                step=1,
                key="inventory_lead_time_days"
            )

        with setting_col2:
            safety_days = st.number_input(
                "Safety Days",
                min_value=0,
                max_value=365,
                value=5,
                step=1,
                key="inventory_safety_days"
            )

        inventory["Lead Time Days"] = int(lead_time_days)
        inventory["Safety Days"] = int(safety_days)

        inventory["Estimated Reorder Point"] = (
            inventory["Average Daily Demand"] *
            (int(lead_time_days) + int(safety_days))
        ).round().astype(int)

        st.markdown("---")
        
        product_options = inventory["Product"].tolist()

        selected_inventory_product = st.selectbox(
            "Select a product",
            product_options,
            key="inventory_product_selector"
        )
                # --------------------------------------------------------
        # MANUAL INVENTORY PLANNING SETTINGS
        # --------------------------------------------------------

        selected_row = inventory[
            inventory["Product"] == selected_inventory_product
        ].iloc[0]

        product_history = df[
            df["Product"] == selected_inventory_product
        ].copy()

        latest_date = df["Date"].max().floor("D")
        start_date = latest_date - pd.Timedelta(days=29)

        product_recent = product_history[
            (product_history["Date"] >= start_date) &
            (product_history["Date"] < latest_date + pd.Timedelta(days=1))
        ]

        daily_product_demand = (
            product_recent.groupby(
                product_recent["Date"].dt.floor("D")
            )["Quantity"]
            .sum()
        )

        full_30_days = pd.date_range(
            start=start_date,
            end=latest_date,
            freq="D"
        )

        daily_product_demand = daily_product_demand.reindex(
            full_30_days,
            fill_value=0
        )

        daily_product_demand.index.name = "Date"

        total_30_day_units = float(selected_row["Total_Quantity"])
        average_daily_demand = float(
            selected_row["Average Daily Demand"]
        )
        reorder_point = int(
            selected_row["Estimated Reorder Point"]
        )

        st.markdown(f"#### {selected_inventory_product}")

        # First row
        a, b = st.columns(2)

        a.metric(
            "Units Sold (Last 30 Days)",
            f"{total_30_day_units:,.0f} units"
        )

        b.metric(
            "Average Daily Demand",
            f"{average_daily_demand:,.2f} units/day"
        )

        # Second row
        c, d = st.columns(2)

        c.metric(
            "Estimated Reorder Point",
            f"{reorder_point:,} units"
        )

        d.metric(
            "Lead Time + Safety",
            f"{int(lead_time_days + safety_days)} days"
        )

        product_chart_data = daily_product_demand.reset_index()
        product_chart_data.columns = ["Date", "Units Sold"]

        fig = px.line(
            product_chart_data,
            x="Date",
            y="Units Sold",
            markers=True,
            labels={
                "Date": "Date",
                "Units Sold": "Units Sold"
            }
        )

        fig.update_traces(
            hovertemplate=(
                "Date: %{x|%d %b %Y}<br>"
                "Units Sold: %{y:,.0f}"
                "<extra></extra>"
            )
        )

        show_chart(
            style_chart(
                fig,
                f"Last 30 Days Demand — {selected_inventory_product}"
            )
        )

        st.caption(
            "Daily values show recorded sales. Days with no recorded "
            "transactions are displayed as zero; this does not necessarily "
            "mean there was no underlying customer demand."
        )

        # --------------------------------------------------------
        # REORDER POINT EXPLANATION
        # --------------------------------------------------------

        st.markdown("### Estimated Reorder Point")

        st.write(
            "Reorder Point = Average Daily Demand × "
            "(Lead Time + Safety Days)"
        )

        st.write(
            f"For **{selected_inventory_product}**: "
            f"{average_daily_demand:.2f} × "
            f"({int(selected_row['Lead Time Days'])} + "
            f"{int(selected_row['Safety Days'])}) = "
            f"**{reorder_point:,} units**"
        )

        st.caption(
            "This estimate uses the latest 30 calendar days, a default "
            "lead time of 7 days and 5 safety days. It is a planning "
            "estimate, not a live stock alert, because stock-on-hand "
            "and purchase orders are not included."
        )

        # --------------------------------------------------------
        # COMPLETE INVENTORY TABLE
        # --------------------------------------------------------

        st.markdown("---")
        st.markdown("### Complete Inventory Overview")

        st.dataframe(
            inventory[
                [
                    "Product",
                    "Total_Quantity",
                    "Average Daily Demand",
                    "Lead Time Days",
                    "Safety Days",
                    "Estimated Reorder Point"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Inventory Report",
            data=inventory.to_csv(index=False).encode("utf-8"),
            file_name="inventory_intelligence.csv",
            mime="text/csv"
        )


# ============================================================
# 23. BUSINESS INSIGHTS
# ============================================================

elif selected_page == "Business Insights":

    show_header(
        "Business Insights",
        "Understand business performance and identify useful trends."
    )

    monthly = get_monthly_sales(df).copy()

    if len(monthly) < 2:

        st.warning("At least two months of data are required.")

    else:

        latest_date = df["Date"].max()
        latest_month = latest_date.to_period("M")

        if latest_date.day < latest_date.days_in_month:

            complete_months = monthly[
                monthly["Date"].dt.to_period("M") < latest_month
            ].copy()

            st.info(
                f"The latest month ({latest_month.strftime('%b %Y')}) "
                "is incomplete and excluded from month-to-month comparison."
            )

        else:

            complete_months = monthly.copy()

        if len(complete_months) < 2:

            st.warning(
                "Not enough complete months are available for comparison."
            )

        else:

            previous = float(complete_months.iloc[-2]["Revenue"])
            latest = float(complete_months.iloc[-1]["Revenue"])

            change = (
                (latest - previous) / previous * 100
                if previous != 0 else 0
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Latest Complete Month",
                complete_months.iloc[-1]["Month"]
            )

            c2.metric(
                "Previous Complete Month",
                complete_months.iloc[-2]["Month"]
            )

            c3.metric(
                "Revenue Change",
                f"{change:+.2f}%"
            )

            st.markdown("### Monthly Revenue Comparison")

            fig = px.line(
                complete_months,
                x="Month",
                y="Revenue",
                markers=True
            )

            fig.update_traces(
                hovertemplate=(
                    "Month: %{x}<br>"
                    "Revenue: %{y:,.2f}"
                    "<extra></extra>"
                )
            )

            show_chart(style_chart(fig, "Complete Monthly Revenue"))

        st.markdown("### Top Revenue-Generating Products")

        top_products = (
            get_product_revenue(df)
            .head(10)
            .sort_values("Revenue")
        )

        fig = px.bar(
            top_products,
            x="Revenue",
            y="Product",
            orientation="h"
        )

        fig.update_traces(
            hovertemplate=(
                "Product: %{y}<br>"
                "Revenue: %{x:,.2f}"
                "<extra></extra>"
            )
        )

        show_chart(style_chart(fig, "Top 10 Products"))

        st.markdown("### Business Observations")

        product_revenue = get_product_revenue(df)

        best_product = product_revenue.iloc[0]["Product"]
        highest_revenue = product_revenue.iloc[0]["Revenue"]

        st.write(f"- **Top revenue product:** {best_product}")
        st.write(f"- **Revenue from top product:** {fmt(highest_revenue)}")
        st.write(f"- **Total unique products:** {df['Product'].nunique():,}")
        st.write(
            f"- **Average order-line revenue:** {fmt(df['Revenue'].mean())}"
        )


# ============================================================
# 24. WHAT-IF BUSINESS SIMULATOR - OPTIMIZED
# ============================================================

elif selected_page == "What-if Simulator":

    show_header(
        "What-if Business Simulator",
        "Explore hypothetical future revenue scenarios using AI forecasts."
    )

    daily_sales = get_daily_sales(df)

    latest_date = daily_sales.index.max()

    first_future_date = latest_date + pd.Timedelta(days=1)
    max_forecast_date = latest_date + pd.Timedelta(days=365)

    st.info(
        f"Historical data ends on **{latest_date.date()}**. "
        f"Maximum forecast date: **{max_forecast_date.date()}**."
    )

    trained = train_sales_model(daily_sales)

    if trained is None:

        st.warning(
            "At least 35 calendar days of historical data are required "
            "for future simulation."
        )

    else:

        model, features, mae, rmse, baseline_mae, evaluation = trained

        period_type = st.selectbox(
            "Select simulation period",
            [
                "Monthly",
                "Yearly",
                "Custom Date Range"
            ]
        )

        selected_start = None
        selected_end = None

        if period_type == "Monthly":

            months = pd.period_range(
                first_future_date.to_period("M"),
                max_forecast_date.to_period("M"),
                freq="M"
            )

            month_options = [str(month) for month in months]

            selected_month = st.selectbox(
                "Select future month",
                month_options
            )

            chosen_period = pd.Period(
                selected_month,
                freq="M"
            )

            selected_start = max(
                chosen_period.start_time.normalize(),
                first_future_date
            )

            selected_end = min(
                chosen_period.end_time.normalize(),
                max_forecast_date
            )

        elif period_type == "Yearly":

            year_options = list(
                range(
                    first_future_date.year,
                    max_forecast_date.year + 1
                )
            )

            selected_year = st.selectbox(
                "Select future year",
                year_options
            )

            selected_start = max(
                pd.Timestamp(
                    year=selected_year,
                    month=1,
                    day=1
                ),
                first_future_date
            )

            selected_end = min(
                pd.Timestamp(
                    year=selected_year,
                    month=12,
                    day=31
                ),
                max_forecast_date
            )

        else:

            col1, col2 = st.columns(2)

            with col1:

                selected_start = st.date_input(
                    "Start Date",
                    value=first_future_date.date(),
                    min_value=first_future_date.date(),
                    max_value=max_forecast_date.date()
                )

            with col2:

                default_end = min(
                    first_future_date + pd.Timedelta(days=29),
                    max_forecast_date
                )

                selected_end = st.date_input(
                    "End Date",
                    value=default_end.date(),
                    min_value=first_future_date.date(),
                    max_value=max_forecast_date.date()
                )

            selected_start = pd.Timestamp(selected_start)
            selected_end = pd.Timestamp(selected_end)

        st.markdown("### Scenario Settings")

        col1, col2 = st.columns(2)

        with col1:

            demand_change = st.slider(
                "Expected Demand Change (%)",
                min_value=-50,
                max_value=100,
                value=10,
                step=1,
                format="%d%%"
            )

        with col2:

            price_change = st.slider(
                "Expected Selling Price Change (%)",
                min_value=-30,
                max_value=50,
                value=0,
                step=1,
                format="%d%%"
            )

        st.caption(
            "Select any whole percentage, including 2%, 3%, 7% or 13%."
        )

        if selected_start > selected_end:

            st.error("Start date cannot be after End Date.")

        elif st.button("Run Business Simulation", type="primary"):

            with st.spinner("Generating future forecast and scenario..."):

                # Calculate only the number of days required.
                days_needed = max(
                    1,
                    (pd.Timestamp(selected_end).normalize() -
                     latest_date.normalize()).days
                )

                all_future = forecast_future(
                    daily_sales,
                    model,
                    features,
                    days_needed
                )

                selected_forecast = all_future[
                    (all_future["Date"] >= selected_start) &
                    (all_future["Date"] <= selected_end)
                ].copy()

                if selected_forecast.empty:

                    st.error(
                        "No forecast data is available for this period."
                    )

                else:

                    baseline_revenue = float(
                        selected_forecast["Predicted Sales"].sum()
                    )

                    demand_factor = 1 + demand_change / 100
                    price_factor = 1 + price_change / 100

                    scenario_revenue = (
                        baseline_revenue *
                        demand_factor *
                        price_factor
                    )

                    revenue_difference = (
                        scenario_revenue - baseline_revenue
                    )

                    difference_percent = (
                        revenue_difference / baseline_revenue * 100
                        if baseline_revenue != 0 else 0
                    )

                    st.markdown("### Simulation Results")

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "AI Forecast Baseline Revenue",
                        fmt(baseline_revenue)
                    )

                    c2.metric(
                        "Scenario Revenue",
                        fmt(scenario_revenue),
                        delta=f"{difference_percent:+.2f}%"
                    )

                    st.metric(
                        "Estimated Revenue Difference",
                        fmt(revenue_difference)
                    )

                    comparison = pd.DataFrame({
                        "Scenario": [
                            "AI Forecast Baseline",
                            "What-if Scenario"
                        ],
                        "Revenue": [
                            baseline_revenue,
                            scenario_revenue
                        ]
                    })

                    fig = px.bar(
                        comparison,
                        x="Scenario",
                        y="Revenue",
                        text_auto=",.2f"
                    )

                    fig.update_traces(
                        hovertemplate=(
                            "Scenario: %{x}<br>"
                            "Revenue: %{y:,.2f}"
                            "<extra></extra>"
                        )
                    )

                    show_chart(
                        style_chart(
                            fig,
                            "Baseline vs Scenario Revenue"
                        )
                    )

                    st.caption(
                        "This is a hypothetical sensitivity simulation, "
                        "not a causal prediction. It applies selected "
                        "demand and price changes to the AI forecast baseline. "
                        "It estimates revenue, not profit."
                    )

                    scenario_report = selected_forecast.copy()

                    scenario_report["Scenario Revenue"] = (
                        scenario_report["Predicted Sales"] *
                        demand_factor *
                        price_factor
                    )

                    st.download_button(
                        "Download Simulation Report",
                        data=scenario_report.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="what_if_simulation.csv",
                        mime="text/csv"
                    )


# ============================================================
# 25. DOWNLOAD REPORTS
# ============================================================

elif selected_page == "Download Reports":

    show_header(
        "Download Business Reports",
        "Export business summaries and your complete cleaned dataset."
    )

    st.markdown("### Report generation")

    st.write(
        "The Excel report contains compact summaries. "
        "The complete cleaned dataset is available separately as CSV "
        "to avoid slow Excel generation for very large datasets."
    )

    with st.spinner("Preparing summary reports..."):

        daily_sales = get_daily_sales(df)

        daily_report = daily_sales.reset_index()
        daily_report.columns = ["Date", "Revenue"]

        monthly_report = get_monthly_sales(df)
        product_report = get_product_revenue(df)
        inventory_report = calculate_inventory(df)

        summary_data = {
            "Daily Sales": daily_report,
            "Monthly Sales": monthly_report,
            "Product Revenue": product_report,
            "Inventory": inventory_report
        }

        excel_file = create_summary_excel(summary_data)

    st.success("Summary reports are ready.")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("### Business Summary Excel")

        st.write(
            "Includes daily sales, monthly sales, product revenue "
            "and inventory intelligence."
        )

        st.download_button(
            "📥 Download Summary Excel",
            data=excel_file,
            file_name="AI_Business_Intelligence_Summary.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            type="primary"
        )

    with c2:

        st.markdown("### Complete Cleaned Dataset")

        st.write(
            f"Includes all {len(df):,} cleaned records in CSV format."
        )

        st.download_button(
            "📥 Download Complete CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_business_data.csv",
            mime="text/csv"
        )

    st.markdown("### Preview: Product Revenue")

    st.dataframe(
        product_report.head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 26. FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.markdown("""
<div style="color: #8795aa; font-size: 11px; line-height: 1.8;">
    AI BUSINESS DECISION INTELLIGENCE<br>
    MACHINE LEARNING • ANALYTICS<br>
    DEMAND FORECASTING
</div>
""", unsafe_allow_html=True)