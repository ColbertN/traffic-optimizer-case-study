from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import TARGET
from .config import POLICY_LABELS


def save_training_monitoring_plots(
    psi_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    train_monitor: pd.DataFrame,
    test_monitor: pd.DataFrame,
    trials_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    # Save all model-training plots into one presentation-friendly folder.
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_psi(psi_df, output_dir / "psi_train_vs_test.png")
    plot_feature_importance(feature_importance, output_dir / "feature_importance.png")
    plot_prediction_distribution(
        train_monitor, test_monitor, output_dir / "train_test_prediction_comparison.png"
    )
    plot_residuals_by_hour(test_monitor, output_dir / "test_residuals_by_hour.png")
    plot_optuna_history(trials_df, output_dir / "optuna_validation_rmse.png")


def save_model_metric_outputs(metadata: dict, output_dir: Path, plot_dir: Path) -> pd.DataFrame:
    """Save a small metric table and plot for the final selected model."""

    # Save metrics as machine-readable CSV and human-friendly Excel.
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = model_metrics_frame(metadata)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    try:
        metrics_df.to_excel(output_dir / "model_metrics.xlsx", index=False)
    except ImportError:
        # Excel export is optional because some environments may not have openpyxl.
        pass

    # Create a compact chart showing held-out test MAE, RMSE, and R2.
    plot_model_test_metrics(metrics_df, plot_dir / "model_test_metrics.png")
    return metrics_df


def save_benchmark_plots(
    sim_results: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    # Save all simulator benchmark plots into the same results plot folder.
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_policy_metrics(metrics, output_dir / "policy_metric_comparison.png")
    plot_queue_timeseries(sim_results, output_dir / "queue_timeseries_by_policy.png")
    plot_intersection_heatmap(sim_results, output_dir / "intersection_queue_heatmap.png")
    plot_failure_impact(sim_results, output_dir / "robot_failure_impact.png")


def model_metrics_frame(metadata: dict) -> pd.DataFrame:
    # Flatten metadata metrics into one row per split.
    rows = []
    for split_name, split_metrics in metadata["metrics"].items():
        rows.append(
            {
                "split": split_name,
                "model": metadata.get("best_model_name", "unknown"),
                "target": metadata["target"],
                "mae": split_metrics["mae"],
                "rmse": split_metrics["rmse"],
                "r2": split_metrics["r2"],
                "best_validation_rmse": metadata["best_validation_rmse"],
                "n_trials": metadata["n_trials"],
            }
        )
    return pd.DataFrame(rows)


def plot_model_test_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    # Focus this chart on the held-out test split because it best reflects future performance.
    test_row = metrics_df[metrics_df["split"] == "test"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # MAE and RMSE show the average and larger-error penalty in vehicle counts.
    error_values = [test_row["mae"], test_row["rmse"]]
    axes[0].bar(["MAE", "RMSE"], error_values, color=["#0E7A5F", "#005A45"])
    axes[0].set_title("Test Error")
    axes[0].set_ylabel("Vehicles")
    for idx, value in enumerate(error_values):
        axes[0].text(idx, value, f"{value:.2f}", ha="center", va="bottom")

    # R2 shows how much of the variation in test demand is explained by the model.
    axes[1].bar(["R2"], [test_row["r2"]], color="#2563EB")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Test Fit")
    axes[1].set_ylabel("R2")
    axes[1].text(0, test_row["r2"], f"{test_row['r2']:.3f}", ha="center", va="bottom")

    model_name = str(test_row["model"]).replace("_", " ").title()
    fig.suptitle(f"Selected Model Test Metrics: {model_name}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_psi(psi_df: pd.DataFrame, output_path: Path) -> None:
    # Show the highest-drift features so monitoring issues are easy to spot.
    df = psi_df.sort_values("psi", ascending=True).tail(15)
    colors = ["#0E7A5F" if value < 0.1 else "#F0B429" if value < 0.25 else "#C2410C" for value in df["psi"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["feature"], df["psi"], color=colors)
    ax.axvline(0.1, color="#F0B429", linestyle="--", linewidth=1, label="Moderate drift")
    ax.axvline(0.25, color="#C2410C", linestyle="--", linewidth=1, label="High drift")
    ax.set_title("Population Stability Index: Train vs Test")
    ax.set_xlabel("PSI")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(feature_importance: pd.DataFrame, output_path: Path) -> None:
    # Plot the strongest model drivers for interview explainability.
    df = feature_importance.head(18).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    if df.empty:
        ax.text(0.5, 0.5, "Feature importance unavailable", ha="center", va="center")
        ax.axis("off")
    else:
        ax.barh(df["feature"], df["importance"], color="#005A45")
        ax.set_title("Top Feature Importances")
        ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_prediction_distribution(
    train_monitor: pd.DataFrame,
    test_monitor: pd.DataFrame,
    output_path: Path,
) -> None:
    # Compare actual demand and predicted demand across train and test periods.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(train_monitor[TARGET], bins=35, alpha=0.65, label="Train actual", color="#005A45")
    axes[0].hist(test_monitor[TARGET], bins=35, alpha=0.55, label="Test actual", color="#F0B429")
    axes[0].set_title("Actual Demand Distribution")
    axes[0].set_xlabel("Vehicles next 15 min")
    axes[0].legend()

    axes[1].hist(train_monitor["prediction"], bins=35, alpha=0.65, label="Train prediction", color="#005A45")
    axes[1].hist(test_monitor["prediction"], bins=35, alpha=0.55, label="Test prediction", color="#2563EB")
    axes[1].set_title("Prediction Distribution")
    axes[1].set_xlabel("Predicted vehicles next 15 min")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_residuals_by_hour(test_monitor: pd.DataFrame, output_path: Path) -> None:
    # Residuals by hour reveal whether the model struggles during peak periods.
    df = test_monitor.copy()
    df["residual"] = df[TARGET] - df["prediction"]
    hourly = (
        df.groupby("hour")["residual"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("hour")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly["hour"], hourly["mean"], marker="o", color="#005A45", label="Mean residual")
    ax.fill_between(
        hourly["hour"],
        hourly["mean"] - hourly["std"].fillna(0),
        hourly["mean"] + hourly["std"].fillna(0),
        color="#0E7A5F",
        alpha=0.18,
        label="+/- 1 std",
    )
    ax.axhline(0, color="#1F2933", linewidth=1)
    ax.set_title("Test Residuals By Hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Actual - predicted vehicles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_optuna_history(trials_df: pd.DataFrame, output_path: Path) -> None:
    # Plot each Optuna trial and the best validation RMSE achieved so far.
    fig, ax = plt.subplots(figsize=(10, 5))
    if "value" in trials_df.columns:
        complete = trials_df[trials_df["value"].notna()].copy()
        ax.plot(complete["number"], complete["value"], marker="o", color="#52616B", label="Trial RMSE")
        ax.plot(
            complete["number"],
            complete["value"].cummin(),
            color="#005A45",
            linewidth=2,
            label="Best so far",
        )
        ax.set_title("Optuna Validation RMSE")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Validation RMSE")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Optuna history unavailable", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_policy_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    # Compare the policies on the two most business-friendly metrics.
    df = metrics.copy()
    df["label"] = df["policy"].map(POLICY_LABELS).fillna(df["policy"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(df["label"], df["avg_wait_minutes_per_vehicle"], color=["#52616B", "#0E7A5F", "#005A45"])
    axes[0].set_title("Average Wait Time")
    axes[0].set_ylabel("Minutes per vehicle")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(df["label"], df["throughput_rate"], color=["#52616B", "#0E7A5F", "#005A45"])
    axes[1].set_title("Throughput Rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_queue_timeseries(sim_results: pd.DataFrame, output_path: Path) -> None:
    # Sum queues across the network so the full-day congestion pattern is visible.
    df = (
        sim_results.groupby(["timestamp", "policy"])["queue_after"]
        .sum()
        .reset_index()
        .sort_values("timestamp")
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    for policy, block in df.groupby("policy"):
        ax.plot(block["timestamp"], block["queue_after"], label=POLICY_LABELS.get(policy, policy), linewidth=2)
    ax.set_title("Network Queue Over The Day")
    ax.set_ylabel("Vehicles still queued")
    ax.set_xlabel("Time")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_intersection_heatmap(sim_results: pd.DataFrame, output_path: Path) -> None:
    # Show which intersections benefit most from adaptive control.
    pivot = sim_results.pivot_table(
        index="intersection_name",
        columns="policy",
        values="queue_after",
        aggfunc="mean",
    )
    pivot = pivot.rename(columns=POLICY_LABELS)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Average Queue By Intersection And Policy")
    fig.colorbar(image, ax=ax, label="Average queued vehicles")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_failure_impact(sim_results: pd.DataFrame, output_path: Path) -> None:
    # Separate failed-robot periods from normal or managed periods.
    df = sim_results.copy()
    df["failure_state"] = np.where(df["robot_status"] == "failed", "Robot failed", "Robot working/managed")
    summary = (
        df.groupby(["policy", "failure_state"])["queue_after"]
        .mean()
        .reset_index()
        .pivot(index="policy", columns="failure_state", values="queue_after")
    )
    summary = summary.rename(index=POLICY_LABELS)
    fig, ax = plt.subplots(figsize=(10, 5))
    summary.plot(kind="bar", ax=ax, color=["#C2410C", "#0E7A5F"])
    ax.set_title("Queue Impact During Robot Failures")
    ax.set_ylabel("Average queued vehicles")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
