# ------------------- Imports -------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Custom modules
from data_gen import generate
from optimization import optimize_price
from model import (
    compute_unit_cost_if_missing,
    add_time_and_roll_features,
    train,
    FEATURES
) 

# ------------------- Dashboard Configuration -------------------
st.set_page_config(page_title="AI Pricing Dashboard", layout="wide")
st.title("💹 AI Pricing Optimization Dashboard")
st.caption("Predict demand from price & competitors, then maximize profit by choosing the best price.")

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("Data")

    uploaded = st.file_uploader("Upload CSV (or leave empty to auto-load synthetic)", type=["csv"])

    if st.button("Load synthetic sample"):
        df = generate(200)
        st.session_state["df"] = df
        st.success("Loaded synthetic dataset.")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["df"] = df
        st.success("CSV uploaded.")

    st.markdown("---")

    st.header("Training & Optimization")
    min_margin = st.number_input("Min margin %", value=5.0, min_value=0.0, step=0.5) / 100.0
    grid_step = st.number_input("Price grid step", value=0.05, min_value=0.01, step=0.01)
    cap_min = st.number_input("Hard min price (0=auto)", value=0.0, min_value=0.0, step=0.1)
    cap_max = st.number_input("Hard max price (0=auto)", value=0.0, min_value=0.0, step=0.1)

# ------------------- Main Flow -------------------
if "df" not in st.session_state:
    st.info("Upload a CSV or click 'Load synthetic sample' in the sidebar to get started.")

else:
    raw_df = st.session_state["df"].copy()

    st.subheader("Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    # ---------------- Product Selection ----------------
    if "product" not in raw_df.columns:
        st.error("❌ Your dataset has no 'product' column. Add it first.")
        st.stop()

    products = raw_df["product"].unique().tolist()

    st.markdown("### Select Product")
    selected_product = st.selectbox("Choose a product", products)

    # Filter data for this product
    df = raw_df[raw_df["product"] == selected_product].copy()

    if df.empty:
        st.error("❌ No data found for selected product.")
        st.stop()

    st.success(f"Product selected: **{selected_product}** ({len(df)} rows)")

    # ---------- Feature Preparation ----------
    work = compute_unit_cost_if_missing(df)
    work = add_time_and_roll_features(work)

    # ---------- Train Model ----------
    st.markdown("### Train model")
    model, mae = train(work)
    st.success(f"Validation MAE: {mae:.2f} units")

    # ---------- Context for Optimization ----------
    latest = work.sort_values("date").iloc[-1].copy()

    st.markdown("### Context (you can tweak)")
    col1, col2, col3, col4 = st.columns(4)

    latest["promo"] = int(col1.selectbox("Promo", options=[0, 1], index=int(latest.get("promo", 0))))
    latest["competitor_price"] = float(col2.number_input("Competitor price", value=float(latest["competitor_price"])))
    latest["unit_cost"] = float(col3.number_input("Unit cost", value=float(latest["unit_cost"])))
    latest["price"] = float(col4.number_input("Current price (reference)", value=float(latest["price"])))

    # ---------- Optimization ----------
    pmin = cap_min if cap_min > 0 else None
    pmax = cap_max if cap_max > 0 else None

    bundle = {"model": model, "features": FEATURES}
    opt_result = optimize_price(
        latest, bundle,
        price_min=pmin,
        price_max=pmax,
        min_margin_pct=min_margin,
        grid_step=grid_step
    )

    best_price = opt_result["best_price"]
    best_sales = opt_result["predicted_sales"]
    best_profit = opt_result["expected_profit"]

    st.markdown("### Recommendation")
    st.metric("💡 Recommended Price", f"{best_price:.2f}")
    st.metric("📦 Predicted Sales", f"{best_sales:.0f} units")
    st.metric("💰 Expected Profit", f"{best_profit:.2f}")

    # ---------- Plots ----------
    prices = np.arange(
        opt_result["price_min_considered"],
        opt_result["price_max_considered"] + 1e-9,
        grid_step
    )

    X_test = pd.DataFrame([latest] * len(prices))
    X_test["price"] = prices
    demand_hat = np.clip(model.predict(X_test[FEATURES]), 0, None)
    profit = (prices - latest["unit_cost"]) * demand_hat

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Predicted demand vs. price**")
        fig1 = px.line(x=prices, y=demand_hat, labels={"x": "Price", "y": "Predicted sales"})
        fig1.add_vline(x=best_price, line_dash="dash", line_color="green")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown("**Profit vs. price**")
        fig2 = px.line(x=prices, y=profit, labels={"x": "Price", "y": "Profit"})
        fig2.add_vline(x=best_price, line_dash="dash", line_color="green")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------- Footer ----------
    st.markdown("---")
    st.caption("Tip: Use the sidebar to change margin, grid step, and price caps. Upload your own CSV anytime.")
