\
import argparse
# u don t need this file if u have reel data , this file results in sales.csv which will be the input of the model
import numpy as np
import pandas as pd
# argparse → Used to parse command-line arguments.

# numpy → For numerical computations, random numbers, and arrays.

# pandas → For tabular data manipulation, creating DataFrames, and saving CSV files.
def generate(n_days: int = 180, seed: int = 7) -> pd.DataFrame:
#     Defines a function generate that creates a synthetic sales dataset.

# n_days → number of days of data to generate (default 180).

# seed → random seed for reproducibility.

# Returns a pandas DataFrame.
    rng = np.random.default_rng(seed)
#     Creates a random number generator using the specified seed.

# Ensures reproducibility of random numbers.
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    # Generates a sequence of daily dates starting from "2025-01-01" for n_days.
    product = "A"
    # Single product "A" for all rows (simplification).

    # Raw materials (toy seasonal patterns + noise)
    wheat = 200 + 12*np.sin(2*np.pi*dates.dayofyear/180) + rng.normal(0, 2, n_days)
    # Simulates wheat price with:

# Base 200

# Seasonal fluctuation (sin function)

# Random noise from normal distribution (mean=0, std=2)
    sugar = 100 + 8*np.cos(2*np.pi*dates.dayofyear/90) + rng.normal(0, 2, n_days)
#     Simulates sugar price with:

# Base 100

# Seasonal cosine pattern

# Random noise

    # Unit cost derived from materials + base
    unit_cost = 1.8 + 0.004*wheat + 0.002*sugar
# Computes unit production cost as a linear combination of raw material prices + base cost 1.8.

    # Promo flag ~12%
    promo = (rng.random(n_days) < 0.12).astype(int)
# 


    # True latent demand (down with price, up with promo/seasonality)
    base_price = 5.5 + 0.4*np.sin(2*np.pi*dates.dayofyear/60) + rng.normal(0, 0.08, n_days)
    # Simulates base selling price with small seasonal fluctuation + random noise.
    true_sales = (
        720
        - 80*base_price
        + 110*promo
        + 20*np.sin(2*np.pi*dates.dayofyear/7)   # weekly
        + rng.normal(0, 25, n_days)
    )
# Simulates true latent demand:

# 720 → base demand

# -80*base_price → sales decrease with price

# +110*promo → promotions boost sales

# +20*sin(...weekly) → weekly seasonality

# rng.normal(0,25) → noise


    # Clip and round
    true_sales = np.clip(true_sales, 0, None)
# Ensures sales are non-negative.

    df = pd.DataFrame({
        "date": dates,
        "product": product,
        "price": np.round(base_price, 2),
        "sales": np.round(true_sales).astype(int),
        "wheat_price": np.round(wheat, 2),
        "sugar_price": np.round(sugar, 2),
        "unit_cost": np.round(unit_cost, 2),
        "promo": promo
    })
#     Creates a pandas DataFrame with all generated columns.

# Rounds prices to 2 decimals, sales to integer.
    return df

def main():
    ap = argparse.ArgumentParser()
    # Creates an argument parser for command-line usage.
    ap.add_argument("--out", type=str, default="data/sales.csv")
    ap.add_argument("--days", type=int, default=180)
#     Adds optional arguments:

# --out → output CSV file path (default "data/sales.csv")

# --days → number of days to generate (default 180)
    args = ap.parse_args()
# Parses command-line arguments into args.
    df = generate(args.days)
    # Calls the generate function with the specified number of days.
    df.to_csv(args.out, index=False)
#     Saves the generated dataset to CSV.

# index=False → don’t save row numbers.
    print(f"Saved synthetic dataset to {args.out} (rows={len(df)})")
# Prints confirmation of saved file and number of rows.
if __name__ == "__main__":
    main()
# Ensures main() runs only if the script is executed directly, not imported.