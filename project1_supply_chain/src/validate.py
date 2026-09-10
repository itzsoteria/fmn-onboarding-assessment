"""
Rolling-origin backtest of the EWMA demand forecast against two naive
baselines, on the 25 established SKUs (new SKUs excluded -- not enough
history to backtest meaningfully).

For each SKU, for each day in the last 30 days of its history:
  - forecast = EWMA(span=14) computed using only data strictly BEFORE that day
  - compare forecast vs actual units_sold that day
Metric: WAPE (Weighted Absolute Percentage Error) = sum(|actual-forecast|) / sum(actual)
  chosen over plain MAPE because a few near-zero-demand days would blow up
  MAPE with divide-by-near-zero; WAPE is the standard supply-chain metric
  for exactly this reason and is easy to explain to a sponsor: "our
  forecast is off by X% of total volume."

Baselines compared against:
  - naive_last_value : yesterday's units_sold
  - naive_7day_avg    : mean of the trailing 7 days
"""
import numpy as np
import pandas as pd
from data_prep import load_and_clean

BACKTEST_DAYS = 30
EWMA_SPAN = 60
MIN_HISTORY_FOR_BACKTEST = 60


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    return float(np.sum(np.abs(actual - forecast)) / np.sum(actual))


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for sku_id, g in df.groupby("sku_id"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < MIN_HISTORY_FOR_BACKTEST:
            continue  # new SKUs: not enough history to backtest fairly

        sales = g["units_sold"].values
        n = len(sales)
        start = n - BACKTEST_DAYS

        for t in range(start, n):
            history = sales[:t]
            actual = sales[t]

            ewma_fc = pd.Series(history).ewm(span=EWMA_SPAN, adjust=False).mean().iloc[-1]
            naive_last = history[-1]
            naive_7d = history[-7:].mean()

            results.append({
                "sku_id": sku_id, "t": t, "actual": actual,
                "ewma_forecast": ewma_fc,
                "naive_last_value": naive_last,
                "naive_7day_avg": naive_7d,
            })

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"method": f"EWMA (span={EWMA_SPAN}) -- our model", "WAPE": wape(results.actual, results.ewma_forecast)},
        {"method": "Naive: yesterday's value", "WAPE": wape(results.actual, results.naive_last_value)},
        {"method": "Naive: trailing 7-day average", "WAPE": wape(results.actual, results.naive_7day_avg)},
    ]).sort_values("WAPE")


if __name__ == "__main__":
    df = load_and_clean("data/project1_supply_chain_demand.csv")
    results = backtest(df)
    summary = summarize(results)
    print(f"Backtested on {results.sku_id.nunique()} established SKUs, "
          f"last {BACKTEST_DAYS} days each ({len(results)} forecast points)\n")
    print(summary.to_string(index=False))
