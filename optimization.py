import numpy as np
import pandas as pd

# optimization.py
def make_price_grid(price_min, price_max, grid_step, max_points=1000):
    # Convert to float and handle NaN
    price_min = float(price_min) if pd.notna(price_min) else 0.0
    price_max = float(price_max) if pd.notna(price_max) else price_min + 1.0
    grid_step = float(grid_step) if pd.notna(grid_step) else 0.05

    # Safety adjustments
    if price_max <= price_min:
        price_max = price_min + max(1.0, 0.1 * price_min)
    if grid_step <= 0:
        grid_step = max(0.01, 0.01 * price_min)

    range_span = price_max - price_min
    if range_span < 1e-6:
        return np.array([price_min])
    
    # If grid_step is larger than range, just return min and max
    if grid_step > range_span:
        return np.array([price_min, price_max])
    
    # Compute number of steps safely
    n_points = int(np.ceil(range_span / grid_step)) + 1
    n_points = max(2, min(n_points, max_points))  # at least 2 points
    
    return np.linspace(price_min, price_max, n_points)

def optimize_price(context_row: pd.Series,
                   bundle: dict,
                   price_min=None,
                   price_max=None,
                   min_margin_pct: float = 0.05,
                   grid_step: float = 0.05) -> dict:

    model = bundle["model"]
    features = bundle["features"]
    unit_cost = float(context_row["unit_cost"])
    competitor = float(context_row.get("competitor_price", np.nan))
    current_price = float(context_row.get("price", np.nan))

    # --- 1) BASE LIMITS ---
    if price_min is None:
        price_min = max(unit_cost * (1 + min_margin_pct), 0.01)

    if price_max is None:
        price_max = price_min + 3.0  # default fallback

    # --- 2) ADD ECONOMIC CONSTRAINTS ---
    # never price above competitor + 0.50
    if not np.isnan(competitor):
        price_max = min(price_max, competitor + 0.50)

    # never jump too far from current price (max +0.50)
    if not np.isnan(current_price):
        price_max = min(price_max, current_price + 0.50)

    # ensure range is valid
    if price_max <= price_min:
        price_max = price_min + 0.10

    # --- 3) Generate candidate grid ---
    prices = make_price_grid(price_min, price_max, grid_step)

    # --- 4) Prepare candidate DataFrame ---
    base = context_row[features].copy()
    candidates = pd.DataFrame([base] * len(prices))
    candidates["price"] = prices

    # --- 5) Predict demand ---
    demand_hat = model.predict(candidates[features])
    demand_hat = np.clip(demand_hat, 0, None)

    # --- 6) Profit ---
    profit = (prices - unit_cost) * demand_hat
    best_idx = int(np.argmax(profit))

    return {
        "best_price": float(np.round(prices[best_idx], 2)),
        "predicted_sales": float(np.round(demand_hat[best_idx], 2)),
        "expected_profit": float(np.round(profit[best_idx], 2)),
        "price_min_considered": float(np.round(prices.min(), 2)),
        "price_max_considered": float(np.round(prices.max(), 2))
    }
