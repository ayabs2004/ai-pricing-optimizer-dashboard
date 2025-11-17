# \
# import argparse
# # u don t need this file if u have reel data , this file results in sales.csv which will be the input of the model
# import numpy as np
# import pandas as pd
# # argparse → Used to parse command-line arguments.

# # numpy → For numerical computations, random numbers, and arrays.

# # pandas → For tabular data manipulation, creating DataFrames, and saving CSV files.
# def generate(n_days: int = 180, seed: int = 7) -> pd.DataFrame:
# #     Defines a function generate that creates a synthetic sales dataset.

# # n_days → number of days of data to generate (default 180).

# # seed → random seed for reproducibility.

# # Returns a pandas DataFrame.
#     rng = np.random.default_rng(seed)
# #     Creates a random number generator using the specified seed.

# # Ensures reproducibility of random numbers.
#     dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
#     # Generates a sequence of daily dates starting from "2025-01-01" for n_days.
#     product = "A"
#     # Single product "A" for all rows (simplification).

#     # Raw materials (toy seasonal patterns + noise)
#     wheat = 200 + 12*np.sin(2*np.pi*dates.dayofyear/180) + rng.normal(0, 2, n_days)
#     # Simulates wheat price with:

# # Base 200

# # Seasonal fluctuation (sin function)

# # Random noise from normal distribution (mean=0, std=2)
#     sugar = 100 + 8*np.cos(2*np.pi*dates.dayofyear/90) + rng.normal(0, 2, n_days)
# #     Simulates sugar price with:

# # Base 100

# # Seasonal cosine pattern

# # Random noise

#     # Unit cost derived from materials + base
#     unit_cost = 1.8 + 0.004*wheat + 0.002*sugar
# # Computes unit production cost as a linear combination of raw material prices + base cost 1.8.

#     # Promo flag ~12%
#     promo = (rng.random(n_days) < 0.12).astype(int)
# # 


#     # True latent demand (down with price, up with promo/seasonality)
#     base_price = 5.5 + 0.4*np.sin(2*np.pi*dates.dayofyear/60) + rng.normal(0, 0.08, n_days)
#     # Simulates base selling price with small seasonal fluctuation + random noise.
#     true_sales = (
#         720
#         - 80*base_price
#         + 110*promo
#         + 20*np.sin(2*np.pi*dates.dayofyear/7)   # weekly
#         + rng.normal(0, 25, n_days)
#     )
# # Simulates true latent demand:

# # 720 → base demand

# # -80*base_price → sales decrease with price

# # +110*promo → promotions boost sales

# # +20*sin(...weekly) → weekly seasonality

# # rng.normal(0,25) → noise


#     # Clip and round
#     true_sales = np.clip(true_sales, 0, None)
# # Ensures sales are non-negative.

#     df = pd.DataFrame({
#         "date": dates,
#         "product": product,
#         "price": np.round(base_price, 2),
#         "sales": np.round(true_sales).astype(int),
#         "wheat_price": np.round(wheat, 2),
#         "sugar_price": np.round(sugar, 2),
#         "unit_cost": np.round(unit_cost, 2),
#         "promo": promo
#     })
# #     Creates a pandas DataFrame with all generated columns.

# # Rounds prices to 2 decimals, sales to integer.
#     return df

# def main():
#     ap = argparse.ArgumentParser()
#     # Creates an argument parser for command-line usage.
#     ap.add_argument("--out", type=str, default="data/sales.csv")
#     ap.add_argument("--days", type=int, default=180)
# #     Adds optional arguments:

# # --out → output CSV file path (default "data/sales.csv")

# # --days → number of days to generate (default 180)
#     args = ap.parse_args()
# # Parses command-line arguments into args.
#     df = generate(args.days)
#     # Calls the generate function with the specified number of days.
#     df.to_csv(args.out, index=False)
# #     Saves the generated dataset to CSV.

# # index=False → don’t save row numbers.
#     print(f"Saved synthetic dataset to {args.out} (rows={len(df)})")
# # Prints confirmation of saved file and number of rows.
# if __name__ == "__main__":
#     main()
# # Ensures main() runs only if the script is executed directly, not imported.








# import pandas as pd

# # -------------------------------
# # 1️⃣ Load datasets
# # -------------------------------
# ingredients = pd.read_csv("Dominos_Ingredients.csv")
# orders = pd.read_csv("Dominos_Orders.csv")

# # -------------------------------
# # 2️⃣ Standardize column names
# # -------------------------------
# ingredients.columns = ingredients.columns.str.lower().str.replace(' ', '_')
# orders.columns = orders.columns.str.lower().str.replace(' ', '_')

# # -------------------------------
# # 3️⃣ Convert datatypes
# # -------------------------------
# # Convert numeric columns
# ingredients['ingredient_cost'] = pd.to_numeric(ingredients['ingredient_cost'], errors='coerce')
# ingredients['quantity'] = pd.to_numeric(ingredients['quantity'], errors='coerce')
# orders['price'] = pd.to_numeric(orders['price'], errors='coerce')
# orders['units_sold'] = pd.to_numeric(orders['units_sold'], errors='coerce')

# # Convert dates
# orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')

# # -------------------------------
# # 4️⃣ Handle missing values
# # -------------------------------
# ingredients['ingredient_cost'].fillna(ingredients['ingredient_cost'].mean(), inplace=True)
# ingredients['quantity'].fillna(0, inplace=True)
# orders['price'].fillna(orders['price'].mean(), inplace=True)
# orders['units_sold'].fillna(0, inplace=True)
# orders.dropna(subset=['order_date'], inplace=True)

# # -------------------------------
# # 5️⃣ Remove duplicates
# # -------------------------------
# ingredients.drop_duplicates(inplace=True)
# orders.drop_duplicates(inplace=True)

# # -------------------------------
# # 6️⃣ Remove negative or unrealistic values
# # -------------------------------
# ingredients = ingredients[(ingredients['ingredient_cost'] >= 0) & (ingredients['quantity'] >= 0)]
# orders = orders[(orders['price'] >= 0) & (orders['units_sold'] >= 0)]

# # -------------------------------
# # 7️⃣ Compute unit cost per product
# # -------------------------------
# ingredients['total_ingredient_cost'] = ingredients['ingredient_cost'] * ingredients['quantity']
# product_cost = ingredients.groupby('product_name')['total_ingredient_cost'].sum().reset_index()
# product_cost.rename(columns={'total_ingredient_cost': 'unit_cost'}, inplace=True)

# # -------------------------------
# # 8️⃣ Merge orders with unit cost
# # -------------------------------
# data = orders.merge(product_cost, on='product_name', how='left')
# data.dropna(subset=['unit_cost'], inplace=True)  # drop unmatched products

# # -------------------------------
# # 9️⃣ Compute revenue
# # -------------------------------
# data['revenue'] = data['price'] * data['units_sold']

# # -------------------------------
# # 🔟 Add extra features for ML
# # -------------------------------
# data['day_of_week'] = data['order_date'].dt.dayofweek
# data['month'] = data['order_date'].dt.month

# # Optional: Add promotion flag if available
# # data['promo'] = ...

# # -------------------------------
# # 1️⃣1️⃣ Final clean check
# # -------------------------------
# print(data.info())
# print(data.head())

# # -------------------------------
# # 1️⃣2️⃣ Save cleaned CSV
# # -------------------------------
# data.to_csv("Dominos_Cleaned.csv", index=False)
# print("✅ Cleaned dataset saved as Dominos_Cleaned.csv")












# # import pandas as pd
# # import numpy as np

# # def generate_dominos_features(cleaned_csv="Dominos_Cleaned.csv", seed: int = 42):
# #     """
# #     Reads the cleaned Dominos dataset and generates a DataFrame ready for your model,
# #     including optional promo flags and simulated competitor prices.
# #     """
# #     rng = np.random.default_rng(seed)
    
# #     # -------------------------------
# #     # 1️⃣ Load cleaned CSV
# #     # -------------------------------
# #     df = pd.read_csv(cleaned_csv)
    
# #     # Ensure correct types
# #     df['unit_cost'] = pd.to_numeric(df['unit_cost'], errors='coerce')
# #     df['price'] = pd.to_numeric(df['price'], errors='coerce')
# #     df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
# #     df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
# #     df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
# #     df.dropna(subset=['unit_cost', 'price', 'sales', 'revenue'], inplace=True)
    
# #     # -------------------------------
# #     # 2️⃣ Add optional promo flag (~12% chance)
# #     # -------------------------------
# #     df['promo'] = (rng.random(len(df)) < 0.12).astype(int)
    
# #     # -------------------------------
# #     # 3️⃣ Simulate competitor prices (±5–15% variation)
# #     # -------------------------------
# #     df['competitor_price'] = df['price'] * (1 + rng.uniform(-0.15, 0.15, size=len(df)))
# #     df['competitor_price'] = df['competitor_price'].round(2)
    
# #     # -------------------------------
# #     # 4️⃣ Adjust sales based on promo & competitor price (optional)
# #     # This mimics demand being higher during promotions and lower if competitors are cheaper
# #     # -------------------------------
# #     df['adjusted_sales'] = df['sales'] + 50*df['promo'] - 30*((df['competitor_price'] < df['price']).astype(int))
# #     df['adjusted_sales'] = df['adjusted_sales'].clip(lower=0).astype(int)
    
# #     # -------------------------------
# #     # 5️⃣ Add extra features
# #     # -------------------------------
# #     df['day_of_week'] = df['date'].dt.dayofweek
# #     df['month'] = df['date'].dt.month
    
# #     # -------------------------------
# #     # 6️⃣ Select final columns for the model
# #     # -------------------------------
# #     model_df = df[[
# #         'date', 'product', 'company', 'unit_cost', 'price',
# #         'competitor_price', 'promo', 'adjusted_sales', 'revenue',
# #         'day_of_week', 'month'
# #     ]].copy()
    
# #     return model_df

# # # -------------------------------
# # # Example usage
# # # -------------------------------
# # if __name__ == "__main__":
# #     df_model = generate_dominos_features("Dominos_Cleaned.csv")
# #     print(df_model.head())
# #     print(f"Dataset ready with {len(df_model)} rows")
# #     df_model.to_csv("Dominos_Model_Ready.csv", index=False)





# ------------------- data_gen.py -------------------
# import pandas as pd
# import numpy as np

# def generate_dominos_features(stock_csv="Dominos_Stock_Data.csv", 
#                               sales_csv="sales.csv", 
#                               seed=42):
#     rng = np.random.default_rng(seed)
    
#     # Load stock dataset
#     stock_df = pd.read_csv(stock_csv)
    
#     # Load sales dataset and clean
#     sales_df = pd.read_csv(sales_csv)
#     date_col = 'Order_Date' if 'Order_Date' in sales_df.columns else 'date'
#     product_col = 'Item' if 'Item' in sales_df.columns else 'product'
#     quantity_col = 'Quantity' if 'Quantity' in sales_df.columns else 'sales'
    
#     # Aggregate sales per day & product
#     sales_agg = sales_df.groupby([date_col, product_col])[quantity_col].sum().reset_index()
#     sales_agg.rename(columns={date_col: 'date', product_col: 'product', quantity_col: 'sales'}, inplace=True)
    
#     # Merge stock_df with sales_agg on date & product
#     df = pd.merge(stock_df, sales_agg, on=['date', 'product'], how='left')
#     df['sales'] = df['sales'].fillna(0).astype(int)
    
#     # Ensure correct dtypes
#     numeric_cols = ['unit_cost', 'price', 'sales', 'revenue']
#     for col in numeric_cols:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#     if 'date' in df.columns:
#         df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     df.dropna(subset=[c for c in numeric_cols if c in df.columns], inplace=True)
    
#     # Promo flag (~12% chance)
#     df['promo'] = (rng.random(len(df)) < 0.12).astype(int)
    
#     # Competitor prices (±15% variation)
#     if 'price' in df.columns:
#         df['competitor_price'] = (df['price'] * (1 + rng.uniform(-0.15, 0.15, len(df)))).round(2)
#     else:
#         df['competitor_price'] = rng.uniform(5, 20, len(df)).round(2)
    
#     # Adjusted sales
#     df['adjusted_sales'] = (
#         df['sales'] + 50 * df['promo'] - 
#         30 * (df['competitor_price'] < df.get('price', df['competitor_price'])).astype(int)
#     ).clip(lower=0).astype(int)
    
#     # Date features
#     if 'date' in df.columns:
#         df['day_of_week'] = df['date'].dt.dayofweek
#         df['month'] = df['date'].dt.month
#     else:
#         df['day_of_week'] = rng.integers(0, 6, len(df))
#         df['month'] = rng.integers(1, 12, len(df))
    
#     # Ensure required columns exist
#     for col in ['product', 'company']:
#         if col not in df.columns:
#             df[col] = 'Unknown'
    
#     # Columns for modeling
#     model_df = df[[
#         'date', 'product', 'company', 'unit_cost', 'price',
#         'competitor_price', 'promo', 'adjusted_sales', 'revenue',
#         'day_of_week', 'month'
#     ]].copy()
    
#     model_df.rename(columns={'adjusted_sales': 'sales'}, inplace=True)
    
#     return model_df

# # -------------------------------
# # 🔹 Unified generate() for dashboard
# # -------------------------------
# def generate(n=200, seed=42):
#     """
#     Return a ready-to-use Dominos DataFrame.
#     `n` and `seed` are kept for backward compatibility, but n is ignored here.
#     """
#     return generate_dominos_features("Dominos_Stock_Data.csv", "sales.csv", seed=seed)
import pandas as pd
import numpy as np


def generate_dominos_features(stock_csv="dominos_stock_data.csv",
                              sales_csv="sales.csv",
                              seed=42):
    rng = np.random.default_rng(seed)

    # -------------------------------
    # 1️⃣ Load the stock dataset
    # -------------------------------
    stock_df = pd.read_csv(stock_csv)

    # Rename and convert date
    stock_df.rename(columns={"Date": "date"}, inplace=True)
    stock_df["date"] = pd.to_datetime(stock_df["date"], errors="coerce")

    # Create base fields from stock data
    stock_df["product"] = "Dominos"  # single product by default
    stock_df["price"] = stock_df["Close"].round(2)
    stock_df["unit_cost"] = (stock_df["Close"] * 0.6).round(2)
    stock_df["company"] = "Dominos"

    # Revenue placeholder (will be replaced after merging)
    stock_df["revenue"] = 0

    # Keep only important columns
    stock_df = stock_df[["date", "product", "company", "price", "unit_cost", "revenue"]]

    # -------------------------------
    # 2️⃣ Load the sales dataset
    # -------------------------------
    sales_df = pd.read_csv(sales_csv)

    # Detect column names in sales.csv
    date_col = "Order_Date" if "Order_Date" in sales_df.columns else "date"
    product_col = "Item" if "Item" in sales_df.columns else "product"
    quantity_col = "Quantity" if "Quantity" in sales_df.columns else "sales"

    # Clean and rename
    sales_df[date_col] = pd.to_datetime(sales_df[date_col], errors="coerce")

    # Aggregate sales per date/product
    sales_agg = (
        sales_df.groupby([date_col, product_col])[quantity_col]
        .sum()
        .reset_index()
        .rename(columns={date_col: "date", product_col: "product", quantity_col: "sales"})
    )

    # -------------------------------
    # 3️⃣ Merge stock + sales
    # -------------------------------
    df = pd.merge(stock_df, sales_agg, on=["date", "product"], how="left")

    df["sales"] = df["sales"].fillna(0).astype(int)

    # Revenue = price × sales
    df["revenue"] = (df["price"] * df["sales"]).round(2)

    # -------------------------------
    # 4️⃣ Add Features
    # -------------------------------

    # Promo flag (~12% chance)
    df["promo"] = (rng.random(len(df)) < 0.12).astype(int)

    # Competitor prices (±15% variation)
    df["competitor_price"] = (
        df["price"] * (1 + rng.uniform(-0.15, 0.15, len(df)))
    ).round(2)

    # Adjusted sales
    df["adjusted_sales"] = (
        df["sales"]
        + 50 * df["promo"]
        - 30 * (df["competitor_price"] < df["price"]).astype(int)
    ).clip(lower=0)

    # Date features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # -------------------------------
    # 5️⃣ Final dataset (for model.py)
    # -------------------------------
    model_df = df[
        [
            "date",
            "product",
            "company",
            "unit_cost",
            "price",
            "competitor_price",
            "promo",
            "adjusted_sales",
            "revenue",
            "day_of_week",
            "month",
        ]
    ].copy()

    model_df.rename(columns={"adjusted_sales": "sales"}, inplace=True)

    return model_df


# -------------------------------
# 🔹 Ready-to-use generator
# -------------------------------
def generate(n=200, seed=42):
    """Return a ready-to-use Dominos dataset for the dashboard."""
    return generate_dominos_features("data/Dominos_Stock_Data.csv", "data/sales.csv", seed=seed)
if __name__ == "__main__":
    df = generate_dominos_features("data/Dominos_Stock_Data.csv", "data/sales.csv")
    df.to_csv("Dominos_Model_Ready.csv", index=False)
    print("Dataset generated → Dominos_Model_Ready.csv")
