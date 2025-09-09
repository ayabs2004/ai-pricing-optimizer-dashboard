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
st.caption("Predict demand from price & raw materials, then maximize profit by choosing the best price.")

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("Data")

    uploaded = st.file_uploader("Upload CSV (or leave empty to auto-load synthetic)", type=["csv"])

    if st.button("Load synthetic sample"):
        df = generate(200)
        # df = pd.read_csv("your_real_sales.csv")if its reel data from ur pc

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

    # ---------- Feature Preparation ----------
    work = compute_unit_cost_if_missing(raw_df)
    work = add_time_and_roll_features(work)

    # ---------- Train Model ----------
    st.markdown("### Train model")
    model, mae = train(work)  # train_df, valid_df optional if you modify train()
    st.success(f"Validation MAE: {mae:.2f} units")

    # ---------- Context for Optimization ----------
    latest = work.sort_values("date").iloc[-1].copy()

    st.markdown("### Context (you can tweak)")
    col1, col2, col3, col4, col5 = st.columns(5)

    latest["promo"] = int(col1.selectbox("Promo", options=[0, 1], index=int(latest.get("promo", 0))))
    latest["wheat_price"] = float(col2.number_input("Raw material 1 price", value=float(latest["wheat_price"])))
    latest["sugar_price"] = float(col3.number_input("Raw material 2 price", value=float(latest["sugar_price"])))
    latest["unit_cost"] = float(col4.number_input("Unit cost", value=float(latest["unit_cost"])))
    latest["price"] = float(col5.number_input("Current price (reference)", value=float(latest["price"])))

    # ---------- Optimization ----------
    pmin = cap_min if cap_min > 0 else None
    pmax = cap_max if cap_max > 0 else None

    bundle = {"model": model, "features": FEATURES}
    opt_result = optimize_price(latest, bundle, price_min=pmin, price_max=pmax,
                                min_margin_pct=min_margin, grid_step=grid_step)

    best_price = opt_result["best_price"]
    best_sales = opt_result["predicted_sales"]
    best_profit = opt_result["expected_profit"]

    st.markdown("### Recommendation")
    st.metric("💡 Recommended Price", f"{best_price:.2f}")
    st.metric("📦 Predicted Sales", f"{best_sales:.0f} units")
    st.metric("💰 Expected Profit", f"{best_profit:.2f}")

    # ---------- Plots ----------
    prices = np.arange(opt_result["price_min_considered"], opt_result["price_max_considered"] + 1e-9, grid_step)
    demand_hat = np.clip(model.predict(pd.DataFrame([latest]*len(prices))[FEATURES]), 0, None)
    profit = (prices - latest["unit_cost"]) * demand_hat

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Predicted demand vs. price**")
        fig1 = px.line(x=prices, y=demand_hat, labels={"x": "Price", "y": "Predicted sales"})
        fig1.add_vline(x=best_price)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown("**Profit vs. price**")
        fig2 = px.line(x=prices, y=profit, labels={"x": "Price", "y": "Profit"})
        fig2.add_vline(x=best_price)
        st.plotly_chart(fig2, use_container_width=True)

    # ---------- Footer ----------
    st.markdown("---")
    st.caption("Tip: Use the sidebar to change margin, grid step, and price caps. Upload your own CSV anytime.")



#     These are inputs for the price optimization part of your dashboard, not the model itself. Let me explain each one clearly:

# 1️⃣ Min margin %

# Value: 5.00 → 5%

# Meaning: The minimum profit margin you want on the product.

# Formula used in code:

# \text{minimum price} = \text{unit cost} \times (1 + \text{min margin %})

# Example:

# Unit cost = 10

# Min margin = 5% → minimum price = 10 × 1.05 = 10.5

# Ensures you don’t sell below a certain profit threshold.

# 2️⃣ Price grid step

# Value: 0.05 → 5 cents

# Meaning: Step size for the price search grid.

# The optimizer tests prices like:

# min price
# ,
# min price + 0.05
# ,
# min price + 0.10
# ,
# …
# min price,min price + 0.05,min price + 0.10,…

# Smaller step → more precise, but slower.

# Bigger step → faster, less precise.

# 3️⃣ Hard min price

# Value: 0.20

# Meaning: Absolute lower bound for price.

# If set to 0 → automatic (based on min margin).

# If set > 0 → price will never go below this value, no matter the optimization.

# 4️⃣ Hard max price

# Value: Not given in your snippet, but could be something like 0 → auto

# Meaning: Absolute upper bound for price.

# If 0 → automatically chosen (heuristic based on current price).

# If set → the optimizer will not go beyond this price.

# ✅ Summary:

# Min margin % → ensures minimum profit per unit

# Price grid step → resolution of price search

# Hard min/max price → enforce absolute bounds

# If you want, I can draw a small diagram showing how these parameters affect the price grid and predicted profit, it makes it very intuitive.

# Do you want me to do that?
