"""
Demand forecasting + inventory risk flagging for the supply chain app.

APPROACH (see README for the full rationale):
  - Per-SKU demand forecast: exponentially-weighted moving average (EWMA,
    span=60 days) of units_sold. Chosen over a heavier model (ARIMA/Prophet)
    because: (a) it's transparent enough to defend live -- "we weight recent
    sales, older days count less" -- (b) it needs no per-SKU hyperparameter
    tuning, which matters with only 25 established SKUs, and (c) backtesting
    (validate.py) showed accuracy keeps improving as the span grows and
    plateaus near 60, meaning demand here is close to stationary -- a
    longer memory window beats a short reactive one. It still adapts if a
    SKU's demand genuinely shifts, unlike a full fixed historical average.
  - Demand volatility: rolling std of daily sales, used for a safety-stock
    buffer -- SKUs with erratic demand get more buffer than steady ones,
    not a single fixed number of days for everyone.
  - Reorder point = forecast_daily_demand * lead_time_days + safety_stock
    safety_stock = z * demand_std * sqrt(lead_time_days)   (z=1.65 -> ~95% service level)
  - Flag logic (two severity tiers for "needs attention", not one flat bucket --
    a SKU that will physically run dry before a fresh order could even land is
    a different problem from one that's merely below its reorder trigger):
        CRITICAL   if days_of_cover < lead_time_days (or stock is already 0)
                   -> will stock out before replenishment arrives even if
                      ordered today
        AT_RISK    if current_stock < reorder_point but days_of_cover >= lead_time
                   -> below the recommended reorder trigger; order now, but
                      won't run dry before a fresh order could land
        OVERSTOCK  if days_of_cover > 3x lead_time_days (tunable per category)
        OK         otherwise
  - New SKUs (< 21 days of history): forecast falls back to the category's
    established-SKU average demand curve, and the flag is always shown
    with a "limited history" confidence flag rather than a hard number,
    because a 12-day trend is not a reliable forecast base.
"""
import numpy as np
import pandas as pd

MIN_DAYS_FOR_OWN_FORECAST = 21
# Backtesting (see validate.py) showed WAPE keeps improving as span grows,
# plateauing around span=60 (close to the full-history mean) -- meaning
# demand here is close to stationary with no material trend, so a longer
# memory window beats a reactive short one. Kept below full-history so the
# forecast can still adapt if a SKU's demand genuinely shifts in production.
EWMA_SPAN = 60
SERVICE_LEVEL_Z = 1.65
OVERSTOCK_MULTIPLE = 2.0  # days_of_cover beyond 2x lead time -> capital tied up for no service-level benefit


def build_sku_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """One row per SKU: latest forecast, risk flag, and the numbers behind it."""
    rows = []
    category_daily_demand = _category_avg_daily_demand(df)

    for sku_id, g in df.groupby("sku_id"):
        g = g.sort_values("date")
        n_days = len(g)
        category = g["category"].iloc[0]
        lead_time = int(g["lead_time_days"].iloc[0])
        current_stock = float(g["closing_stock"].iloc[-1])
        low_history = n_days < MIN_DAYS_FOR_OWN_FORECAST

        if low_history:
            forecast_demand = category_daily_demand.get(category, g["units_sold"].mean())
            demand_std = g["units_sold"].std(ddof=0) if n_days > 1 else forecast_demand * 0.5
            basis = f"category average ({category}) -- only {n_days} days of own history"
        else:
            ewma = g["units_sold"].ewm(span=EWMA_SPAN, adjust=False).mean()
            forecast_demand = float(ewma.iloc[-1])
            demand_std = float(g["units_sold"].tail(30).std(ddof=0))
            basis = f"{EWMA_SPAN}-day EWMA of own sales history ({n_days} days available)"

        safety_stock = SERVICE_LEVEL_Z * demand_std * np.sqrt(lead_time)
        reorder_point = forecast_demand * lead_time + safety_stock
        days_of_cover = current_stock / forecast_demand if forecast_demand > 0 else np.inf

        will_run_dry_first = np.isfinite(days_of_cover) and days_of_cover < lead_time
        if current_stock <= 0 or will_run_dry_first:
            flag = "CRITICAL"
        elif current_stock < reorder_point:
            flag = "AT_RISK"
        elif days_of_cover > OVERSTOCK_MULTIPLE * lead_time:
            flag = "OVERSTOCK"
        else:
            flag = "OK"

        rows.append({
            "sku_id": sku_id,
            "category": category,
            "lead_time_days": lead_time,
            "current_stock": current_stock,
            "forecast_daily_demand": round(forecast_demand, 1),
            "demand_volatility": round(float(demand_std), 1),
            "safety_stock": round(float(safety_stock), 1),
            "reorder_point": round(float(reorder_point), 1),
            "days_of_cover": round(float(days_of_cover), 1) if np.isfinite(days_of_cover) else None,
            "flag": flag,
            "low_history": low_history,
            "forecast_basis": basis,
            "history_days": n_days,
        })

    flag_order = {"CRITICAL": 0, "AT_RISK": 1, "OVERSTOCK": 2, "OK": 3}
    snapshot = pd.DataFrame(rows)
    snapshot["_sort"] = snapshot["flag"].map(flag_order)
    snapshot = snapshot.sort_values(
        by=["_sort", "days_of_cover"], ascending=[True, True]
    ).drop(columns="_sort").reset_index(drop=True)
    return snapshot


def _category_avg_daily_demand(df: pd.DataFrame) -> dict:
    established = df.groupby("sku_id").filter(lambda g: len(g) >= MIN_DAYS_FOR_OWN_FORECAST)
    return established.groupby("category")["units_sold"].mean().to_dict()


if __name__ == "__main__":
    from data_prep import load_and_clean
    df = load_and_clean("data/project1_supply_chain_demand.csv")
    snap = build_sku_snapshot(df)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(snap.to_string())
    print()
    print(snap["flag"].value_counts())
