from __future__ import annotations

import argparse
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
from traffic_sa.modeling import feature_importance_frame, train_with_optuna
from traffic_sa.monitoring import save_model_metric_outputs, save_training_monitoring_plots


def parse_args() -> argparse.Namespace:
    # Keep Optuna search size and reproducibility configurable from the command line.
    parser = argparse.ArgumentParser(description="Train the traffic demand model with Optuna.")
    parser.add_argument("--trials", type=int, default=5, help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    # Read command-line options such as Optuna trials and random seed.
    args = parse_args()

    # Training expects the processed table created by scripts/generate_data.py.
    training_path = DATA_PROCESSED_DIR / "model_training_data.csv"
    if not training_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {training_path}. Run scripts/generate_data.py first."
        )

    # Create output folders for the selected model, metrics, and plots.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_RESULTS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Load timestamp as a date column because the model uses a time-based split.
    df = pd.read_csv(training_path, parse_dates=["timestamp"])

    # These paths are passed into the training routine and may be renamed by best model type.
    model_path = MODEL_DIR / "best_traffic_demand_pipeline.joblib"
    metadata_path = MODEL_RESULTS_DIR / "best_model_metadata.json"
    trials_path = MODEL_RESULTS_DIR / "optuna_trials.csv"

    # Run Optuna across feature sets and candidate models, then train the final winner.
    result = train_with_optuna(
        df,
        model_path=model_path,
        metadata_path=metadata_path,
        trials_path=trials_path,
        n_trials=args.trials,
        random_state=args.seed,
    )

    # Extract interpretability and drift-monitoring tables from the fitted model.
    feature_importance = feature_importance_frame(result["artifact"])
    psi_df = result["psi"]

    # Save tabular outputs used for the presentation and code walkthrough.
    feature_importance.to_csv(MODEL_RESULTS_DIR / "feature_importance.csv", index=False)
    psi_df.to_csv(MODEL_RESULTS_DIR / "psi_train_vs_test.csv", index=False)
    result["trials"].to_csv(MODEL_RESULTS_DIR / "optuna_trials.csv", index=False)
    result["train_monitor"].to_csv(MODEL_RESULTS_DIR / "train_predictions.csv", index=False)
    result["test_monitor"].to_csv(MODEL_RESULTS_DIR / "test_predictions.csv", index=False)

    # Save model monitoring plots such as PSI, feature importance, residuals, and Optuna history.
    save_training_monitoring_plots(
        psi_df=psi_df,
        feature_importance=feature_importance,
        train_monitor=result["train_monitor"],
        test_monitor=result["test_monitor"],
        trials_df=result["trials"],
        output_dir=MODEL_RESULTS_IMAGE_DIR,
    )

    # Save final train/test metrics as CSV, Excel, and a presentation-friendly plot.
    metrics_df = save_model_metric_outputs(
        metadata=result["metadata"],
        output_dir=MODEL_RESULTS_DIR,
        plot_dir=MODEL_RESULTS_IMAGE_DIR,
    )

    # Print the most important outputs so the user can confirm the run completed.
    print(f"Saved best model pipeline: {result['metadata']['model_path']}")
    print(f"Saved model metadata: {metadata_path}")
    print(f"Saved model metrics: {MODEL_RESULTS_DIR / 'model_metrics.csv'}")
    print(f"Saved Optuna trials: {trials_path}")
    print(f"Test metrics: {metrics_df[metrics_df['split'] == 'test'].to_dict('records')[0]}")
    print(f"Model results: {MODEL_RESULTS_DIR}")


if __name__ == "__main__":
    main()
