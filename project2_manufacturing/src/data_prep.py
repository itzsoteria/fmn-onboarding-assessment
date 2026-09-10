"""
Data cleaning for the manufacturing sensor dataset.

Known issues in the raw file (found during EDA) that this module fixes:
  1. 20 fully-identical duplicate rows (10 machine/timestamp pairs appear
     twice with identical values) -- appended out of timestamp order near
     the end of the file. Safe to drop_duplicates() since they're exact
     matches, not conflicting readings for the same key.
  2. temperature_c and vibration_mm_s have scattered missing readings
     (648 and 432 rows). These are short sensor dropouts, not structural
     gaps, so they're filled with per-machine linear interpolation on the
     hourly time index -- NOT a global mean (which would erase each
     machine's own baseline, and baselines differ meaningfully machine to
     machine, e.g. average vibration ranges from ~0.30 to ~1.26 mm/s
     across the fleet).
"""
import pandas as pd


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    before = len(df)
    df = df.drop_duplicates()
    n_dupes_dropped = before - len(df)

    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    for col in ["temperature_c", "vibration_mm_s"]:
        df[col] = df.groupby("machine_id")[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )

    df.attrs["n_duplicates_dropped"] = n_dupes_dropped
    return df


def data_quality_report(raw_path: str) -> dict:
    raw = pd.read_csv(raw_path)
    cleaned = load_and_clean(raw_path)
    return {
        "rows_before": len(raw),
        "rows_after": len(cleaned),
        "duplicates_dropped": cleaned.attrs["n_duplicates_dropped"],
        "missing_temperature_before": int(raw["temperature_c"].isna().sum()),
        "missing_vibration_before": int(raw["vibration_mm_s"].isna().sum()),
        "missing_after_cleaning": int(cleaned[["temperature_c", "vibration_mm_s"]].isna().sum().sum()),
        "n_machines": cleaned["machine_id"].nunique(),
        "total_failure_events": int(cleaned["failure_event"].sum()),
    }


if __name__ == "__main__":
    report = data_quality_report("data/project2_manufacturing_sensors.csv")
    for k, v in report.items():
        print(f"{k}: {v}")
