\
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#aleady exists in dashboard
from sklearn.metrics import mean_absolute_error
# argparse → Parse command-line arguments for data path, model output, etc.

# json → Save a simple report (like MAE and features) to a JSON file.

# joblib → Save/load ML models efficiently.

# numpy → For numerical operations and rounding.

# pandas → Data manipulation (DataFrames).

# RandomForestRegressor → ML model for predicting demand (non-linear regression).

# mean_absolute_error → Metric to evaluate prediction accuracy.
FEATURES = [
    "price", "promo",
    "wheat_price", "sugar_price", "unit_cost",
    "month", "dow",
    "sales_lag1", "price_lag1", "sales_ma7", "price_ma7"
]
# Defines columns used as features for the model.

# Includes price, promotion, raw material costs, time features, and rolling stats.

def compute_unit_cost_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    if "unit_cost" not in df.columns:
        # Adjust weights/base for your business reality
        df["unit_cost"] = 0.003*df["wheat_price"].astype(float) + 0.002*df["sugar_price"].astype(float) + 1.8
    return df
#     Checks if unit_cost exists.

# If not, computes it using wheat and sugar prices + base cost 1.8.

# Returns updated DataFrame.


def add_time_and_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["product", "date"]).reset_index(drop=True)
    d["month"] = d["date"].dt.month
    d["dow"] = d["date"].dt.dayofweek
# Converts date to datetime.

# Sorts by product and date.

# Adds month and day-of-week (dow) features.
    # Lags / rolling by product
    d["sales_lag1"] = d.groupby("product")["sales"].shift(1)
    d["price_lag1"] = d.groupby("product")["price"].shift(1)
    d["sales_ma7"]  = d.groupby("product")["sales"].shift(1).rolling(7).mean()
    d["price_ma7"]  = d.groupby("product")["price"].shift(1).rolling(7).mean()
# Computes lag and rolling features:

# Previous day's sales/price (_lag1)

# 7-day moving averages (_ma7)
    # Fill initial NaNs for simplicity
    for col in ["sales_lag1", "price_lag1", "sales_ma7", "price_ma7"]:
        d[col] = d[col].fillna(method="bfill").fillna(method="ffill")
    if "promo" not in d.columns:
        d["promo"] = 0
    return d
# Fills missing lag/rolling values using backfill → forward fill.

# Adds promo column if missing, default 0.
def train(df: pd.DataFrame):
    # Time-based split (last 20% validation)
    df = df.sort_values(["product", "date"])
    split_idx = int(len(df)*0.8)
    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()
# Sorts data by product & date.

# Uses 80% for training, last 20% for validation (time-based split).
    X_train, y_train = train_df[FEATURES], train_df["sales"]
    X_valid, y_valid = valid_df[FEATURES], valid_df["sales"]
# Separates features (X) and target (y) for train and validation sets.
    model = RandomForestRegressor(
        n_estimators=400, max_depth=12, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
#     Creates Random Forest regressor:

# 400 trees, max depth 12, fixed random seed, parallel processing.

# Trains model on X_train/y_train.
    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)

    return model, mae
# Predicts sales on validation set.

# Computes Mean Absolute Error (MAE).

# Returns trained model and MAE.
def main():
    ap = argparse.ArgumentParser(description="Train demand model")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--model", type=str, default="models/demand_model.joblib")
    ap.add_argument("--report", type=str, default="models/report.json")
    args = ap.parse_args()
# Creates argument parser:

# --data → path to CSV dataset (required)

# --model → path to save model

# --report → path to save MAE report
    df = pd.read_csv(args.data)
    # Loads CSV dataset.
    # Basic checks
    for col in ["date", "product", "price", "sales"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if "unit_cost" not in df.columns and (("wheat_price" not in df.columns) or ("sugar_price" not in df.columns)):
        raise ValueError("Provide unit_cost or both wheat_price and sugar_price")
# Validates that required columns exist.

# Raises errors if critical info is missing.
    df = compute_unit_cost_if_missing(df)
    df = add_time_and_roll_features(df)
    df = df.dropna(subset=["unit_cost"] + FEATURES + ["sales"])
# Computes missing unit costs, adds features, and removes rows with missing values.
    model, mae = train(df)
# Trains model and gets validation MAE.
    payload = {
        "model_type": "RandomForestRegressor",
        "features": FEATURES,
        "mae": float(np.round(mae, 3))
     }
    # Saves model and feature list for later prediction.
    # save model and report
    joblib.dump({"model": model, "features": FEATURES}, args.model)
    # Saves model and feature list for later prediction.
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
# Writes JSON report with MAE and features.
    print(f"Saved model to {args.model}")
    print(f"Validation MAE: {mae:.2f} units")
    print(f"Report saved to {args.report}")
# Prints confirmations for model save, MAE, and report.
if __name__ == "__main__":
    main()













# 1. Lag features
# d["sales_lag1"] = d.groupby("product")["sales"].shift(1)
# d["price_lag1"] = d.groupby("product")["price"].shift(1)


# .shift(1) → moves the data down by 1 row within each product group.

# sales_lag1 → previous day’s sales for the same product.

# price_lag1 → previous day’s price for the same product.

# Why useful?

# Many time series are auto-correlated: today’s sales often depend on yesterday’s sales.

# Including the previous day’s price can help the model understand price elasticity over time.

# 2. Rolling (moving average) features
# d["sales_ma7"]  = d.groupby("product")["sales"].shift(1).rolling(7).mean()
# d["price_ma7"]  = d.groupby("product")["price"].shift(1).rolling(7).mean()


# .rolling(7).mean() → computes the average over the last 7 days.

# sales_ma7 → average sales of the past week (excluding today, because of .shift(1)).

# price_ma7 → average price of the past week.

# Why useful?

# Smooths out daily fluctuations or noise.

# Captures trends: e.g., is sales increasing or decreasing over the past week?

# Helps the model learn patterns over time, not just isolated days.

# 3. Handling missing values
# for col in ["sales_lag1", "price_lag1", "sales_ma7", "price_ma7"]:
#     d[col] = d[col].fillna(method="bfill").fillna(method="ffill")


# .fillna(method="bfill") → fills missing values by looking forward (next available value).

# .fillna(method="ffill") → fills remaining missing values by looking backward.

# Why? The first few rows will have NaNs because lag or rolling averages don’t exist yet. Filling ensures no missing values are fed to the model.