"""
AI/ML Sales Forecasting Project
--------------------------------
End-to-end workflow:
1. Load and validate data
2. Clean / preprocess
3. Exploratory analysis
4. Feature engineering for time-series forecasting
5. Train baseline + ML regression models
6. Evaluate model performance
7. Forecast future monthly sales

This project uses a small synthetic sales dataset so it can run immediately.
Replace sales_data.csv with a real dataset while keeping the same columns.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path(__file__).with_name("sales_data.csv")


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"date", "sales", "promotion", "ad_spend"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "sales"]).copy()
    df = df.drop_duplicates().sort_values("date").reset_index(drop=True)

    # Fill any missing numeric values conservatively.
    for col in ["sales", "promotion", "ad_spend"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Time-based features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Lag and rolling features (only using past information)
    df["lag_1"] = df["sales"].shift(1)
    df["lag_2"] = df["sales"].shift(2)
    df["rolling_mean_3"] = df["sales"].shift(1).rolling(3).mean()

    return df.dropna().reset_index(drop=True)


def explore_data(df: pd.DataFrame) -> None:
    print("\n=== DATA OVERVIEW ===")
    print(df.head())
    print("\nRows:", len(df))
    print("\nMissing values:\n", df.isna().sum())
    print("\nSummary statistics:\n", df[["sales", "promotion", "ad_spend"]].describe())

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["sales"], marker="o")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("sales_trend.png"), dpi=150)
    plt.close()


def evaluate(name: str, model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred, squared=False)
    metrics = {
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": rmse,
        "R2": r2_score(y_test, pred),
    }
    return metrics


def train_models(df: pd.DataFrame):
    features = [
        "promotion",
        "ad_spend",
        "year",
        "month",
        "quarter",
        "lag_1",
        "lag_2",
        "rolling_mean_3",
    ]

    X = df[features]
    y = df["sales"]

    # Time-aware split: first 80% for training, latest 20% for testing
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=42,
        ),
    }

    results = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))
        fitted[name] = model

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]
    print(f"\nBest model based on RMSE: {best_name}")

    return best_model, features


def forecast_next_month(df: pd.DataFrame, model, features):
    last = df.iloc[-1]
    next_date = (last["date"] + pd.offsets.MonthBegin(1)).normalize()

    # Simple scenario assumptions for the next month
    promotion = 0
    ad_spend = float(df["ad_spend"].tail(6).mean())

    row = pd.DataFrame([{
        "promotion": promotion,
        "ad_spend": ad_spend,
        "year": next_date.year,
        "month": next_date.month,
        "quarter": next_date.quarter,
        "lag_1": float(df["sales"].iloc[-1]),
        "lag_2": float(df["sales"].iloc[-2]),
        "rolling_mean_3": float(df["sales"].tail(3).mean()),
    }])[features]

    prediction = float(model.predict(row)[0])

    print("\n=== NEXT-MONTH FORECAST ===")
    print(f"Forecast month: {next_date.strftime('%Y-%m')}")
    print(f"Predicted sales: {prediction:,.0f}")

    return next_date, prediction


def main():
    df = load_and_prepare_data(DATA_PATH)
    explore_data(df)
    model, features = train_models(df)
    forecast_next_month(df, model, features)


if __name__ == "__main__":
    main()
