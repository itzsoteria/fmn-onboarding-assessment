"""
Data cleaning for the supply chain demand dataset.

Known issues in the raw file (found during EDA) that this module fixes:
  1. `category` has inconsistent casing (e.g. "Snacks" vs "SNACKS").
  2. `units_sold` and `closing_stock` have scattered missing values.
     These are NOT filled with a generic forward-fill: the data obeys an
     exact inventory identity wherever it isn't missing --
         closing_stock[t] = closing_stock[t-1] - units_sold[t] + units_received[t]
     -- so we solve for the missing side of that equation first, and only
     fall back to interpolation for the rare rows where neither side is
     recoverable (e.g. missing value on day 1, or two missing values in a row).
"""
import pandas as pd
import numpy as np


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["category"] = df["category"].str.strip().str.title()
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    df = _impute_via_flow_balance(df)

    # Final safety net: any value the identity couldn't recover
    # (e.g. missing on the very first day of a SKU's history) gets a
    # per-SKU linear interpolation rather than being left null.
    for col in ["units_sold", "closing_stock"]:
        df[col] = df.groupby("sku_id")[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )

    df["units_sold"] = df["units_sold"].round(0)
    df["closing_stock"] = df["closing_stock"].round(0)
    return df


def _impute_via_flow_balance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for sku, idx in df.groupby("sku_id").groups.items():
        idx = list(idx)
        for pos in range(1, len(idx)):
            i, i_prev = idx[pos], idx[pos - 1]
            sold, recv, close = df.at[i, "units_sold"], df.at[i, "units_received"], df.at[i, "closing_stock"]
            prev_close = df.at[i_prev, "closing_stock"]

            if pd.isna(close) and pd.notna(sold) and pd.notna(prev_close):
                df.at[i, "closing_stock"] = prev_close - sold + recv
            elif pd.isna(sold) and pd.notna(close) and pd.notna(prev_close):
                df.at[i, "units_sold"] = prev_close + recv - close
    return df


def data_quality_report(raw_path: str) -> dict:
    """Quick before/after summary, useful for the README / model summary doc."""
    raw = pd.read_csv(raw_path)
    cleaned = load_and_clean(raw_path)
    return {
        "rows": len(raw),
        "n_skus": raw["sku_id"].nunique(),
        "category_case_variants_found": raw["category"].nunique() - cleaned["category"].nunique(),
        "missing_units_sold_before": int(raw["units_sold"].isna().sum()),
        "missing_closing_stock_before": int(raw["closing_stock"].isna().sum()),
        "missing_after_cleaning": int(cleaned[["units_sold", "closing_stock"]].isna().sum().sum()),
    }


if __name__ == "__main__":
    report = data_quality_report("data/project1_supply_chain_demand.csv")
    for k, v in report.items():
        print(f"{k}: {v}")
