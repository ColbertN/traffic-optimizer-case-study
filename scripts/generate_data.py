from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_sa.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODEL_RESULTS_DIR
from traffic_sa.simulation import GautengTrafficDataGenerator, intersection_configs_as_frame


def parse_args() -> argparse.Namespace:
    # Keep the data-generation run configurable from the command line.
    parser = argparse.ArgumentParser(description="Generate synthetic Gauteng traffic data.")
    parser.add_argument("--days", type=int, default=45, help="Number of synthetic days to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    # Read command-line options such as number of days and random seed.
    args = parse_args()

    # Ensure the expected project folders exist before saving generated files.
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate raw 15-minute traffic observations and convert them into model-ready rows.
    generator = GautengTrafficDataGenerator(seed=args.seed)
    raw_df = generator.generate_raw_observations(days=args.days)
    training_df = generator.make_training_table(raw_df)

    # Keep raw observations separate from the processed training table.
    raw_path = DATA_RAW_DIR / "gauteng_traffic_observations.csv"
    training_path = DATA_PROCESSED_DIR / "model_training_data.csv"
    intersections_path = MODEL_RESULTS_DIR / "intersection_config.csv"

    # Save all generated outputs so training and demo scripts can reuse them.
    raw_df.to_csv(raw_path, index=False)
    training_df.to_csv(training_path, index=False)
    intersection_configs_as_frame().to_csv(intersections_path, index=False)

    # Print the key file locations for the presenter running the pipeline.
    print(f"Saved raw observations: {raw_path} ({len(raw_df):,} rows)")
    print(f"Saved model training data: {training_path} ({len(training_df):,} rows)")
    print(f"Saved intersection config: {intersections_path}")


if __name__ == "__main__":
    main()
