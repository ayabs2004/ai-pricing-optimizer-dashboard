\
import numpy as np
import pandas as pd
#difernce between this , optimize_price file and the other that exists in dashboard
import joblib
# numpy → For numerical computations and creating price grids.

# pandas → For creating DataFrames of candidate prices.

# joblib → Not used in this snippet, but usually for loading/saving models.
def optimize_price(context_row: pd.Series,
                   bundle: dict,
                   price_min=None, price_max=None,
                   min_margin_pct: float = 0.05,
                   grid_step: float = 0.05) -> dict:
#     context_row: a single row of features (like a current product-day record) without the price.

# bundle: dictionary containing:

# "model" → trained ML model

# "features" → list of feature column names

# price_min / price_max → optional bounds for price search.

# min_margin_pct → ensures price is at least unit_cost * (1 + min_margin_pct).

# grid_step → step size for price search.

# Returns a dictionary with best price, predicted sales, and expected profit.
    """
    context_row: a single row with all feature columns except 'price' (we will grid-search price).
    bundle: {'model': trained_model, 'features': feature_list}
    """
    model = bundle["model"]
    features = bundle["features"]
    unit_cost = float(context_row["unit_cost"])
# Retrieves trained model and feature list from the bundle.

# Gets the unit cost for this product/context.
    # Bounds
    if price_min is None:
        price_min = unit_cost * (1 + min_margin_pct)
    # If no minimum price specified, set minimum price as unit_cost + margin.
    if price_max is None:
        # heuristic upper bound
        price_max = max(price_min + 2.0, context_row.get("price", price_min) + 3.0)
# If no maximum price specified, set a heuristic upper bound: either 2 above min or 3 above current price.
    prices = np.arange(price_min, price_max + 1e-9, grid_step)
#     Creates array of candidate prices from price_min to price_max with step grid_step.

# 1e-9 ensures the price_max is included.
    base = context_row[features].copy()
    # Copies feature values of the context row to use as a template for all candidate prices.
    # candidate table
    candidates = pd.DataFrame([base]*len(prices))
    candidates["price"] = prices
# Creates a DataFrame where each row has the same features except for the price, which is set to each candidate value.
    demand_hat = model.predict(candidates[features])
    demand_hat = np.clip(demand_hat, 0, None)
    # Uses the trained model to predict sales for each candidate price.

# Clips predictions to be non-negative.
    profit = (prices - unit_cost) * demand_hat
# Calculates profit for each price:

# Profit
# =
# (
# Price
# −
# Unit Cost
# )
# ×
# Predicted Sales
# Profit=(Price−Unit Cost)×Predicted Sales
    best_idx = int(np.argmax(profit))
    # Finds the index of the price that gives maximum profit.
    return {
        "best_price": float(np.round(prices[best_idx], 2)),
        "predicted_sales": float(np.round(demand_hat[best_idx], 2)),
        "expected_profit": float(np.round(profit[best_idx], 2)),
        "price_min_considered": float(np.round(prices.min(), 2)),
        "price_max_considered": float(np.round(prices.max(), 2))
    }
# Returns a dictionary with:

# "best_price" → price that maximizes profit

# "predicted_sales" → sales predicted at that price

# "expected_profit" → profit at that price

# "price_min_considered" → lowest price in grid

# "price_max_considered" → highest price in grid




# Summary:

# Creates a grid of possible prices based on cost and optional bounds.

# Copies the context features for each candidate price.

# Predicts demand for each price.

# Computes profit.

# Returns best price and associated metrics.