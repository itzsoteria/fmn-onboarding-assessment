"""
Feature engineering for machine failure risk.

Features, all computed per-machine (never mixed across machines, since
baselines genuinely differ -- e.g. average vibration ranges from ~0.30 to
~1.26 mm/s across the fleet):
  - roll_mean_temp_24h, roll_std_temp_24h   : short-term level & noise
  - roll_mean_vib_24h,  roll_std_vib_24h
  - roll_mean_vib_72h                       : longer-term trend for comparison
  - temp_z, vib_z                           : deviation from THIS machine's
                                               own full-history mean/std --
                                               this is what actually travels
                                               well to new machines with a
                                               different baseline, and what
                                               grounds the LLM explanation
                                               ("2.3 std above its own normal")
  - vib_trend_24h                            : roll_mean_vib_24h minus its
                                               value 24h ago -- is vibration
                                               actively climbing right now?
  - run_hours_since_maintenance              : as given

Label: failure_within_24h = 1 if a failure_event occurs anywhere in the
NEXT 24 hours for that machine. This reframes 17 point-in-time failure
events into a ~24x larger, still-honest positive class -- "will this
machine fail soon" rather than "will it fail in this exact hour", which
is both what the business actually wants and more learnable from limited
data. First 72h of each machine's history are dropped (rolling windows
aren't meaningful yet) -- consistent with treating short-history machines
as a distinct cold-start case.
"""
import numpy as np
import pandas as pd

MIN_HOURS_FOR_FEATURES = 72
LOOKAHEAD_HOURS = 24


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for machine_id, g in df.groupby("machine_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)

        g["roll_mean_temp_24h"] = g["temperature_c"].rolling(24, min_periods=24).mean()
        g["roll_std_temp_24h"] = g["temperature_c"].rolling(24, min_periods=24).std()
        g["roll_mean_vib_24h"] = g["vibration_mm_s"].rolling(24, min_periods=24).mean()
        g["roll_std_vib_24h"] = g["vibration_mm_s"].rolling(24, min_periods=24).std()
        g["roll_mean_vib_72h"] = g["vibration_mm_s"].rolling(72, min_periods=72).mean()

        # Deviation from this machine's OWN baseline (computed once, from its
        # full history) -- what makes the signal transferable across machines
        # with different normal operating ranges.
        temp_mu, temp_sigma = g["temperature_c"].mean(), g["temperature_c"].std()
        vib_mu, vib_sigma = g["vibration_mm_s"].mean(), g["vibration_mm_s"].std()
        g["temp_z"] = (g["temperature_c"] - temp_mu) / temp_sigma
        g["vib_z"] = (g["vibration_mm_s"] - vib_mu) / vib_sigma

        g["vib_trend_24h"] = g["roll_mean_vib_24h"] - g["roll_mean_vib_24h"].shift(24)

        # Label: does a failure happen anywhere in the next LOOKAHEAD_HOURS?
        future_failure = g["failure_event"][::-1].rolling(LOOKAHEAD_HOURS, min_periods=1).max()[::-1]
        g["failure_within_24h"] = future_failure.shift(-1).fillna(0).astype(int)

        g["baseline_temp_mean"] = temp_mu
        g["baseline_vib_mean"] = vib_mu

        out.append(g)

    result = pd.concat(out, ignore_index=True)
    result = result[result.groupby("machine_id").cumcount() >= MIN_HOURS_FOR_FEATURES].reset_index(drop=True)
    return result


FEATURE_COLUMNS = [
    "roll_mean_temp_24h", "roll_std_temp_24h",
    "roll_mean_vib_24h", "roll_std_vib_24h", "roll_mean_vib_72h",
    "temp_z", "vib_z", "vib_trend_24h",
    "run_hours_since_maintenance",
]


if __name__ == "__main__":
    from data_prep import load_and_clean
    df = load_and_clean("data/project2_manufacturing_sensors.csv")
    feats = build_features(df)
    print("Rows after feature engineering:", len(feats))
    print("Positive label rate:", feats["failure_within_24h"].mean().round(4))
    print(feats[["machine_id", "timestamp"] + FEATURE_COLUMNS + ["failure_within_24h"]].tail(10))
