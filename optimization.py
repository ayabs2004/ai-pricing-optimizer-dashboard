import numpy as np
import pandas as pd
import joblib

# numpy → For numerical computations and creating price grids.
# pandas → For creating DataFrames of candidate prices.
# joblib → (Optional) For loading/saving trained models.


def optimize_price(context_row: pd.Series,
                   bundle: dict,
                   price_min=None,
                   price_max=None,
                   min_margin_pct: float = 0.05,
                   grid_step: float = 0.05) -> dict:
    """
    Perform grid search to find the profit-maximizing price.

    Parameters
    ----------
    context_row : pd.Series
        A single row with all feature columns (including price, promo, etc.).
    bundle : dict
        Contains:
            - 'model' → trained ML model
            - 'features' → list of feature column names
    price_min, price_max : float, optional
        Optional bounds for price search.
    min_margin_pct : float, default 0.05
        Ensures minimum price ≥ unit_cost × (1 + min_margin_pct)
    grid_step : float, default 0.05
        Step size for price search.

    Returns
    -------
    dict
        {
            "best_price": float,
            "predicted_sales": float,
            "expected_profit": float,
            "price_min_considered": float,
            "price_max_considered": float
        }
    """

    # Retrieve model and feature list from the bundle
    model = bundle["model"]
    features = bundle["features"]

    # Extract unit cost from the current context
    unit_cost = float(context_row["unit_cost"])

    # Determine price bounds
    if price_min is None:
        price_min = unit_cost * (1 + min_margin_pct)  # ensure minimum profit margin
    if price_max is None:
        # heuristic upper bound (auto-adjusts if not specified)
        price_max = max(price_min + 2.0, context_row.get("price", price_min) + 3.0)

    # Create a range of candidate prices
    prices = np.arange(price_min, price_max + 1e-9, grid_step)

    # Duplicate the current context for each price
    base = context_row[features].copy()
    candidates = pd.DataFrame([base] * len(prices))
    candidates["price"] = prices

    # Predict demand for each candidate price
    demand_hat = model.predict(candidates[features])
    demand_hat = np.clip(demand_hat, 0, None)  # ensure non-negative sales

    # Compute expected profit
    profit = (prices - unit_cost) * demand_hat

    # Find the best price
    best_idx = int(np.argmax(profit))

    # Return optimization results
    return {
        "best_price": float(np.round(prices[best_idx], 2)),
        "predicted_sales": float(np.round(demand_hat[best_idx], 2)),
        "expected_profit": float(np.round(profit[best_idx], 2)),
        "price_min_considered": float(np.round(prices.min(), 2)),
        "price_max_considered": float(np.round(prices.max(), 2))
    }
