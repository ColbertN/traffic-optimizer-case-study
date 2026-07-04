from __future__ import annotations

import numpy as np
import pandas as pd


# Core time, location, and current traffic-pressure fields.
BASE_NUMERIC_FEATURES = [
    "hour",
    "minute",
    "hour_float",
    "day_of_week",
    "is_weekend",
    "is_school_day",
    "intersection_id",
    "lanes",
    "base_volume",
    "current_vehicles",
    "current_queue_length",
    "taxi_ratio",
    "non_compliance_rate",
]

# Rain features capture slower driving and longer clearance time during wet conditions.
WEATHER_FEATURES = ["is_rain", "rain_intensity"]

# Reliability features capture South African constraints such as robot failures and pointsmen.
RELIABILITY_FEATURES = [
    "load_shedding_stage",
    "special_event",
    "robot_failed",
    "pointsman_available",
    "incident_flag",
]

# Lag features tell the model what happened in the previous 15 minutes and previous hour.
LAG_FEATURES = [
    "vehicles_lag_15m",
    "vehicles_lag_1h",
    "queue_lag_15m",
    "rolling_1h_vehicles",
    "rolling_1h_queue",
]

# Categorical fields are one-hot encoded inside the preprocessing pipeline.
BASE_CATEGORICAL_FEATURES = ["direction", "corridor", "area_type"]
RELIABILITY_CATEGORICAL_FEATURES = ["robot_status"]

# The supervised-learning target is demand in the next 15-minute interval.
TARGET = "target_vehicles_next_15m"


def build_feature_list(
    use_weather: bool = True,
    use_reliability: bool = True,
    use_lag_features: bool = True,
    use_robot_status: bool = True,
) -> tuple[list[str], list[str]]:
    # Start with the always-on feature groups.
    numeric = list(BASE_NUMERIC_FEATURES)
    categorical = list(BASE_CATEGORICAL_FEATURES)

    # Optuna can switch these groups on or off to find the best signal set.
    if use_weather:
        numeric.extend(WEATHER_FEATURES)
    if use_reliability:
        numeric.extend(RELIABILITY_FEATURES)
    if use_lag_features:
        numeric.extend(LAG_FEATURES)
    if use_robot_status:
        categorical.extend(RELIABILITY_CATEGORICAL_FEATURES)

    return numeric, categorical


def time_based_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    valid_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Sort by time so the test set represents future periods, not random leakage.
    ordered = df.sort_values("timestamp").copy()
    unique_times = ordered["timestamp"].drop_duplicates().sort_values().to_numpy()

    # Split on timestamps so every direction/intersection at the same time stays together.
    train_end = int(len(unique_times) * train_fraction)
    valid_end = int(len(unique_times) * (train_fraction + valid_fraction))

    train_times = set(unique_times[:train_end])
    valid_times = set(unique_times[train_end:valid_end])

    train_df = ordered[ordered["timestamp"].isin(train_times)].copy()
    valid_df = ordered[ordered["timestamp"].isin(valid_times)].copy()
    test_df = ordered[~ordered["timestamp"].isin(train_times | valid_times)].copy()
    return train_df, valid_df, test_df


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate PSI for numeric train-vs-test drift monitoring."""

    # Coerce to numeric and ignore missing values before binning.
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return 0.0

    # Use train quantiles as the expected distribution boundaries.
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) <= 2:
        breakpoints = np.linspace(expected.min(), expected.max() + 1e-6, bins + 1)

    # Compare the percentage of train and test observations inside each bin.
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Clip tiny percentages to avoid divide-by-zero and log-of-zero errors.
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def psi_table(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
) -> pd.DataFrame:
    # Calculate PSI feature by feature for monitoring plots and CSV output.
    rows = []
    for feature in numeric_features:
        if feature in train_df.columns and feature in test_df.columns:
            rows.append(
                {
                    "feature": feature,
                    "psi": population_stability_index(train_df[feature], test_df[feature]),
                }
            )
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
