import pandas as pd
import numpy as np

def generate(sales_csv="data/sales.csv", seed=42):
    rng = np.random.default_rng(seed)

    # -------------------------------
    # 1️⃣ Load & aggregate sales
    # -------------------------------
    sales_df = pd.read_csv(sales_csv)
    sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")

    df = (
        sales_df.groupby(["date", "product"])
        .agg(
            sales=("sales", "sum"),
            unit_cost=("unit_cost", "mean"),
            promo=("promo", "max")
        )
        .reset_index()
    )

    # Add synthetic raw material prices if not present
    if 'wheat_price' not in df.columns:
        df['wheat_price'] = rng.uniform(0.5, 1.5, len(df))
    if 'sugar_price' not in df.columns:
        df['sugar_price'] = rng.uniform(0.5, 1.5, len(df))

    # -------------------------------
    # 2️⃣ Time features
    # -------------------------------
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # -------------------------------
    # 3️⃣ PRICE ESTIMATION (pricing)
    # -------------------------------

    # Cost-plus pricing
    margin = rng.uniform(0.35, 0.6, len(df))
    base_price = df["unit_cost"] * (1 + margin)

    # Raw material pressure
    raw_index = (
        df["wheat_price"] + df["sugar_price"]
    ) / (df["wheat_price"].mean() + df["sugar_price"].mean())

    raw_factor = 1 + 0.2 * (raw_index - 1)

    # Seasonality
    seasonal_factor = np.where(df["month"].isin([6,7,8]), 1.05, 1.0)

    # Promo discount
    promo_factor = np.where(df["promo"] == 1, 0.9, 1.0)

    # Estimated price
    df["estimated_price"] = (
        base_price
        * raw_factor
        * seasonal_factor
        * promo_factor
    ).round(2)

    # -------------------------------
    # 4️⃣ Synthetic competitor price
    # -------------------------------
    df["competitor_price"] = (
        df["estimated_price"]
        * (1 + rng.uniform(-0.15, 0.15, len(df)))
    ).round(2)

    # -------------------------------
    # 5️⃣ SALES ESTIMATION (demand)
    # -------------------------------

    adjusted_sales = df["sales"].astype(float)

    # Price elasticity
    price_elasticity = -0.8
    adjusted_sales *= (
        df["estimated_price"] / df["estimated_price"].mean()
    ) ** price_elasticity

    # Promo impact (proportionnel)
    promo_strength = rng.uniform(0.1, 0.3, len(df))
    adjusted_sales *= (1 + promo_strength * df["promo"])

    # Competition impact
    price_gap = (df["competitor_price"] - df["estimated_price"]) / df["estimated_price"]
    adjusted_sales *= (1 + 0.7 * price_gap.clip(upper=0))

    # Weekend boost
    weekend = df["day_of_week"].isin([4,5,6]).astype(int)
    adjusted_sales *= (1 + 0.15 * weekend)

    # Product sensitivity
    product_factor = {
        "Pizza": 1.2,
        "Drink": 0.8,
        "Dessert": 0.9
    }
    adjusted_sales *= df["product"].map(product_factor).fillna(1.0)

    df["estimated_sales"] = adjusted_sales.clip(lower=0).round(0)

    # -------------------------------
    # 6️⃣ Revenue
    # -------------------------------
    df["revenue"] = (df["estimated_price"] * df["estimated_sales"]).round(2)

    # -------------------------------
    # 7️⃣ Final dataset
    # -------------------------------
    df["company"] = "Dominos"

    model_df = df[[
        "date", "product", "company",
        "unit_cost", "estimated_price", "competitor_price",
        "promo", "estimated_sales",
        "revenue", "day_of_week", "month",
        "wheat_price", "sugar_price"
    ]].rename(columns={
        "estimated_price": "price",
        "estimated_sales": "sales"
    })

    return model_df
def compute_unit_cost(product, date, bom, raw_prices):
    rows = bom[bom["product"] == product]
    cost = 0.0
    for _, r in rows.iterrows():
        price = raw_prices.loc[
            (raw_prices["raw_material"] == r["raw_material"]) &
            (raw_prices["date"] <= date),
            "price"
        ].iloc[-1]
        cost += r["qty_per_unit"] * price
    return cost
