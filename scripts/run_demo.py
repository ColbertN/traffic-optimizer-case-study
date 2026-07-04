from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_sa.config import (
    DATA_PROCESSED_DIR,
    MODEL_DIR,
    MODEL_RESULTS_DIR,
    MODEL_RESULTS_IMAGE_DIR,
)
from traffic_sa.modeling import load_model_artifact
from traffic_sa.monitoring import save_benchmark_plots
from traffic_sa.simulation import GautengTrafficSimulator


def parse_args() -> argparse.Namespace:
    # Allow a specific simulation date, or automatically choose a high-pressure day.
    parser = argparse.ArgumentParser(description="Run the traffic-control benchmark demo.")
    parser.add_argument(
        "--date",
        default=None,
        help="Optional date from the generated data, e.g. 2026-06-10. Default picks a high-pressure day.",
    )
    return parser.parse_args()


def main() -> None:
    # Read optional demo date from the command line.
    args = parse_args()

    # The demo needs the processed data table plus the trained model metadata.
    data_path = DATA_PROCESSED_DIR / "model_training_data.csv"
    metadata_path = MODEL_RESULTS_DIR / "best_model_metadata.json"
    model_path = resolve_model_path(metadata_path)

    # Stop early with clear instructions if data or model artifacts are missing.
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run scripts/generate_data.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run scripts/train_model.py first.")

    # Load generated observations and the saved preprocessing/model pipeline.
    raw_df = pd.read_csv(data_path, parse_dates=["timestamp"])
    artifact = load_model_artifact(model_path)

    # Pick the demo day and filter the data down to that day only.
    demo_date = args.date or choose_demo_date(raw_df)
    demo_df = raw_df[raw_df["date"] == demo_date].copy()
    if demo_df.empty:
        raise ValueError(f"No generated observations found for date {demo_date}.")

    # Compare fixed, adaptive, and failure-aware robot policies on the same traffic day.
    simulator = GautengTrafficSimulator()
    sim_results, metrics = simulator.run_benchmark(demo_df, model_pipeline=artifact["pipeline"])

    # Ensure all folders exist before saving the demo outputs.
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_RESULTS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Save both detailed simulation rows and the policy-level summary metrics.
    sim_path = DATA_PROCESSED_DIR / "demo_simulation_results.csv"
    metrics_path = MODEL_RESULTS_DIR / "demo_policy_metrics.csv"
    sim_results.to_csv(sim_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    # Save visual plots used in the Part B presentation.
    save_benchmark_plots(sim_results, metrics, MODEL_RESULTS_IMAGE_DIR)
    sim_results.to_csv(MODEL_RESULTS_DIR / "demo_simulation_results.csv", index=False)

    # Calculate the headline benefit of failure-aware control versus fixed robots.
    fixed_wait = metrics.loc[metrics["policy"] == "fixed", "avg_wait_minutes_per_vehicle"].iloc[0]
    best_wait = metrics.loc[
        metrics["policy"] == "failure_aware", "avg_wait_minutes_per_vehicle"
    ].iloc[0]
    improvement = 100 * (fixed_wait - best_wait) / max(fixed_wait, 1e-6)

    # Print the demo results so they are visible without opening the CSV files.
    print(f"Demo date: {demo_date}")
    print(f"Saved simulation results: {sim_path}")
    print(f"Saved policy metrics: {metrics_path}")
    print(metrics.to_string(index=False))
    print(f"Failure-aware wait-time improvement vs fixed: {improvement:.1f}%")
    print(f"Model results: {MODEL_RESULTS_DIR}")


def resolve_model_path(metadata_path: Path) -> Path:
    # Prefer the exact model path recorded during training.
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_path = Path(metadata["model_path"])
        if model_path.exists():
            return model_path

    # Fall back to the newest named best-model file in the models folder.
    candidates = sorted(MODEL_DIR.glob("best_*_traffic_demand_pipeline.joblib"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    # Final fallback keeps older runs compatible with the original generic filename.
    return MODEL_DIR / "best_traffic_demand_pipeline.joblib"


def choose_demo_date(raw_df: pd.DataFrame) -> str:
    # Score each day by demand pressure, failed robots, and incidents.
    scoring_df = raw_df[(raw_df["hour_float"] >= 5.5) & (raw_df["hour_float"] <= 18.5)].copy()
    daily = (
        scoring_df.groupby("date")
        .agg(
            total_vehicles=("current_vehicles", "sum"),
            failed_periods=("robot_failed", "sum"),
            incidents=("incident_flag", "sum"),
        )
        .reset_index()
    )

    # Higher scores represent a more interesting day for the policy benchmark.
    daily["score"] = (
        daily["total_vehicles"]
        + 250 * daily["failed_periods"]
        + 120 * daily["incidents"]
    )
    return str(daily.sort_values("score", ascending=False).iloc[0]["date"])


if __name__ == "__main__":
    main()
