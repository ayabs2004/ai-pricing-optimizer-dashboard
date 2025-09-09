# AI Finance Project — Price Optimization from Raw Material Costs + Sales History

This mini-project generates synthetic sales data, trains a demand model, and optimizes selling price to maximize profit.
It also includes a **Streamlit dashboard** for interactive exploration.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Generate synthetic data

```bash
python src/data_gen.py --out data/sales.csv
```

### 2) Train the demand model

```bash
python src/model.py --data data/sales.csv --model models/demand_model.joblib --report models/report.json
```

### 3) Run the dashboard

```bash
streamlit run src/dashboard.py
```

The app lets you load CSV (your own or the synthetic one), train the model, and get an **optimal price** with charts.

### CSV schema

- `date` (YYYY-MM-DD)
- `product` (string)
- `price` (float) — selling price
- `sales` (int) — units sold
- `wheat_price` (float) (wheat as an example of a first raw material)
- `sugar_price` (float) (sugar as an example of a second raw material)
- `unit_cost` (float) — per-unit variable cost (if missing, the code can compute it)
- `promo` (0/1)

You can add more columns (e.g., competitor price) and then include them as features in `src/model.py` and `src/dashboard.py`.
