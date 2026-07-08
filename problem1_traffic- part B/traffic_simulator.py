"""
Problem 1: Smart City Traffic Management

This file keeps the original starter-code entry point, but delegates the real
implementation to src/traffic_sa. The implementation is South Africa-specific:
Gauteng commuter peaks, unreliable robots, pointsman fallback, taxi behaviour,
and failure-aware adaptive signal control.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_sa.simulation import GautengTrafficDataGenerator, GautengTrafficSimulator


def main() -> None:
    """Run a quick local simulator smoke test."""

    generator = GautengTrafficDataGenerator(seed=42)
    raw_df = generator.generate_raw_observations(days=2)

    simulator = GautengTrafficSimulator()
    sim_results, metrics = simulator.run_benchmark(raw_df)

    print("Failure-Aware Gauteng Traffic Simulator")
    print("=" * 45)
    print(f"Generated observations: {len(raw_df):,}")
    print(f"Simulation records: {len(sim_results):,}")
    print(metrics.to_string(index=False))
    print("\nFor the full workflow run:")
    print("python scripts/generate_data.py")
    print("python scripts/train_model.py")
    print("python scripts/run_demo.py")


if __name__ == "__main__":
    main()
