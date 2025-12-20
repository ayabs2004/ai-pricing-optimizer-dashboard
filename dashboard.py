# ------------------- Imports -------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#import matplotlib.pyplot as plt
#import seaborn as sns


from data_gen import generate
from optimization import make_price_grid, optimize_price
from model import (
    compute_unit_cost_if_missing,
    add_time_and_roll_features,
    train,
    FEATURES
)
# def evaluate_model(model, X, y, display_plot=True):
#     """
#     Evaluate a regression model with multiple metrics and optional plots.
#     
#     Parameters:
#         model : trained regression model
#         X     : features (DataFrame)
#         y     : true target values (Series)
#         display_plot : whether to plot results
#     
#     Returns:
#         metrics_dict : dict with MAE, RMSE, R², MAPE
#     """
#     y_pred = model.predict(X)

#     # Metrics
#     mae = mean_absolute_error(y, y_pred)
#     rmse = np.sqrt(mean_squared_error(y, y_pred))
#     r2 = r2_score(y, y_pred)
#     # Avoid division by zero for MAPE
#     mask = y != 0
#     mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100

#     metrics_dict = {
#         "MAE": mae,
#         "RMSE": rmse,
#         "R2": r2,
#         "MAPE": mape
#     }

#     if display_plot:
#         # Actual vs Predicted
#         fig1, ax1 = plt.subplots()
#         ax1.scatter(y, y_pred, alpha=0.7)
#         ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
#         ax1.set_xlabel("Actual Sales")
#         ax1.set_ylabel("Predicted Sales")
#         ax1.set_title("Actual vs Predicted Sales")
#         st.pyplot(fig1)

#         # Residuals
#         residuals = y - y_pred
#         fig2, ax2 = plt.subplots()
#         sns.histplot(residuals, kde=True, bins=30, ax=ax2)
#         ax2.set_title("Residuals Distribution")
#         st.pyplot(fig2)

#     return metrics_dict
# ------------------- Page config -------------------
st.set_page_config(page_title="AI Pricing Dashboard", layout="wide")
st.title("💹 AI Pricing Optimization Dashboard")
st.caption("Predict demand from price & competitors, then maximize profit.")

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("Data")

    uploaded = st.file_uploader("Upload CSV (or auto-load synthetic)", type=["csv"])

    if st.button("Load synthetic sample"):
        st.session_state["df"] = generate()
        st.success("Synthetic dataset loaded.")

    if uploaded is not None:
        st.session_state["df"] = pd.read_csv(uploaded)
        st.success("CSV uploaded.")

    st.markdown("---")
    st.header("Training & Optimization")

    min_margin = st.number_input("Min margin %", 5.0) / 100
    grid_step = st.number_input("Price grid step", value=0.05, min_value=0.01)
    cap_min = st.number_input("Hard min price (0=auto)", 0.0)
    cap_max = st.number_input("Hard max price (0=auto)", 0.0)

# ------------------- Main -------------------
if "df" not in st.session_state:
    st.info("Upload a CSV or load a synthetic dataset.")
    st.stop()

raw_df = st.session_state["df"].copy()
st.subheader("Preview")
st.dataframe(raw_df.head(20), use_container_width=True)

# ---------------- Product selection ----------------
if "product" not in raw_df.columns:
    st.error("Dataset must contain a 'product' column.")
    st.stop()

product = st.selectbox("Choose product", raw_df["product"].unique())
df = raw_df[raw_df["product"] == product].copy()

# ---------------- Feature prep ----------------
work = compute_unit_cost_if_missing(df)
work = add_time_and_roll_features(work)

# Safety


work = work.dropna(subset=FEATURES + ["sales"])

# ---------------- Train ----------------

model, mae,extra = train(work)
y_true = work["sales"]
X = work[FEATURES]

# metrics = evaluate_model(model, X, y_true)
# st.success(
#     f"Validation MAE: {metrics['MAE']:.2f} units | "
#     f"RMSE: {metrics['RMSE']:.2f} | "
#     f"R²: {metrics['R2']:.2f} | "
#     f"MAPE: {metrics['MAPE']:.2f}%"
# )




# ---------------- Optimization context ----------------
latest = work.sort_values("date").iloc[-1].copy()

for col, default in {
    "promo": 0,
    "price": latest["price"],
    "unit_cost": latest["unit_cost"],
    "competitor_price": latest["price"]
}.items():
    if col not in latest or pd.isna(latest[col]):
        latest[col] = default

st.markdown("### Context")
c1, c2, c3, c4 = st.columns(4)

latest["promo"] = c1.selectbox("Promo", [0, 1], index=int(latest["promo"]))
latest["competitor_price"] = c2.number_input("Competitor price", value=float(latest["competitor_price"]))
latest["unit_cost"] = c3.number_input("Unit cost", value=float(latest["unit_cost"]))
latest["price"] = c4.number_input("Reference price", value=float(latest["price"]))

# ---------------- Optimization ----------------
unit_cost = latest["unit_cost"]

pmin = cap_min if cap_min > 0 else max(unit_cost * (1 + min_margin), latest["price"] * 0.6)
pmax = cap_max if cap_max > 0 else latest["price"] * 1.8

bundle = {"model": model, "features": FEATURES}

opt = optimize_price(
    latest,
    bundle,
    price_min=pmin,
    price_max=pmax,
    min_margin_pct=min_margin,
    grid_step=grid_step
)

best_price = opt["best_price"]

st.markdown("### Recommendation")
st.metric("💡 Optimal Price", f"{best_price:.2f}")
st.metric("📦 Predicted Sales", f"{opt['predicted_sales']:.0f}")
st.metric("💰 Expected Profit", f"{opt['expected_profit']:.2f}")

# ---------------- Plots ----------------
prices = make_price_grid(opt["price_min_considered"], opt["price_max_considered"], grid_step)

X_test = pd.DataFrame([latest] * len(prices))
X_test["price"] = prices

demand = np.clip(model.predict(X_test[FEATURES]), 0, None)
profit = (prices - unit_cost) * demand

c1, c2 = st.columns(2)

with c1:
    fig = px.line(x=prices, y=demand, labels={"x": "Price", "y": "Predicted Sales"})
    fig.add_vline(x=best_price, line_dash="dash", line_color="green")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.line(x=prices, y=profit, labels={"x": "Price", "y": "Profit"})
    fig.add_vline(x=best_price, line_dash="dash", line_color="green")
    st.plotly_chart(fig, use_container_width=True)
