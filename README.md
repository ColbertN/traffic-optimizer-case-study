# Failure-Aware Adaptive Robot Control For Gauteng Peak Traffic
Humbulani Colbert Nekhumbe
nekhumbecolbert3@gmail.com
079 724 4438
Linkedin: www.linkedin.com/in/humbulani-colbert-nekhumbe-12266b123

Guthub: https://github.com/ColbertN

Part B technical case-study solution for **Problem 1: Smart Traffic Light Optimization**.

This project adapts the supplied traffic-light starter code into a South African context: Gauteng commuter peaks, unreliable robots, pointsman fallback, taxi behaviour, rain, load-shedding proxy variables, and limited public-transport alternatives that increase private-car pressure.

## Problem Framing

In South Africa, peak-hour traffic is not only caused by too many cars. It is also worsened by:

- morning inbound commuter flow from residential areas into JHB CBD, Sandton, Midrand, Fourways, and Pretoria;
- afternoon outbound flow back home;
- robots that fail because of power, vandalism, maintenance, or cable issues;
- informal pointsmen/civilians trying to keep traffic moving;
- taxis and some drivers not fully complying with robot control;
- rain, incidents, and limited public-transport coverage.

The solution compares:

1. **Fixed robots**: same timing regardless of demand.
2. **Adaptive robots**: ML predicts near-term demand and optimization adjusts green splits.
3. **Failure-aware adaptive control**: detects robot failure pressure, assigns pointsman-style fallback, and optimizes surrounding flow.

## Project Structure

```text
Technical CaseStudy Code/
  data/
    raw/                         synthetic Gauteng traffic observations
    processed/                   model-ready data and demo simulation outputs
  models/
    best_<model_name>_traffic_demand_pipeline.joblib
  model_results/                 model metrics, MLOps checks, and demo outputs
    best_model_metadata.json
    demo_policy_metrics.csv
    demo_simulation_results.csv
    feature_importance.csv
    intersection_config.csv
    model_metrics.csv
    model_metrics.xlsx
    optuna_trials.csv
    psi_train_vs_test.csv
    test_predictions.csv
    train_predictions.csv
    plots/
  scripts/
    generate_data.py             creates synthetic Gauteng traffic data
    train_model.py               tunes, trains, evaluates, and saves the best model
    run_demo.py                  runs the policy benchmark using the saved model
  src/traffic_sa/
    __init__.py                  marks traffic_sa as the project package
    config.py                    central paths, intersections, and policy labels
    features.py                  feature lists, time split, and PSI drift logic
    modeling.py                  preprocessing, Optuna tuning, and model persistence
    monitoring.py                metrics tables and plot generation
    simulation.py                synthetic data generator and traffic-policy simulator
  main.py                        simple entry point for the saved demo benchmark
  requirements.txt               Python dependencies
```

## Python File Responsibilities

- `main.py`: runs the saved demo by delegating to `scripts/run_demo.py`.
- `scripts/generate_data.py`: creates raw synthetic traffic observations and the processed model-training table.
- `scripts/train_model.py`: runs Optuna across candidate models and feature groups, then saves the best pipeline and model results.
- `scripts/run_demo.py`: loads the saved model and compares fixed, adaptive, and failure-aware robot policies.
- `src/traffic_sa/__init__.py`: package marker so project modules can be imported cleanly.
- `src/traffic_sa/config.py`: stores reusable folder paths, intersection definitions, robot-status factors, and chart labels.
- `src/traffic_sa/features.py`: defines model features, the target variable, time-based train/test split, and PSI drift monitoring.
- `src/traffic_sa/modeling.py`: builds the preprocessing pipeline, tunes models with Optuna, calculates metrics, and saves the model artifact.
- `src/traffic_sa/monitoring.py`: creates model-result tables and presentation plots.
- `src/traffic_sa/simulation.py`: generates South African traffic data and simulates how each robot-control policy affects queues and wait time.

## How To Run

From this folder:

```bash
python scripts/generate_data.py --days 45
python scripts/train_model.py --trials 5
python scripts/run_demo.py
```

Or run the saved demo directly:

```bash
python main.py
```

## Current Results

Using the saved model and demo day selected by the script:

| Policy | Avg wait minutes/vehicle | Throughput rate | Max queue |
|---|---:|---:|---:|
| Fixed robots | 76.344 | 0.8555 | 2367.46 |
| Adaptive robots | 60.621 | 0.8892 | 1334.21 |
| Failure-aware adaptive | 58.382 | 0.8950 | 1287.55 |

**Failure-aware adaptive control improves average wait time by 23.5% vs fixed robots.**

Model test performance:

- MAE: 9.6697 vehicles
- RMSE: 14.8540 vehicles
- R2: 0.9446

## ML Component

The model predicts **vehicles expected in the next 15 minutes** for each intersection and direction.

Features include time of day, intersection, corridor, current vehicles, queue length, taxi ratio, non-compliance proxy, robot failure, pointsman availability, incidents, load-shedding proxy, and lag/rolling traffic features.

Optuna tunes feature groups, model family, and model hyperparameters. Only the best preprocessing + model pipeline is saved in `models/`, and the filename includes the winning model family:

```text
models/best_extra_trees_traffic_demand_pipeline.joblib
```

## Optimization Component

Every 15-minute interval, the simulator predicts demand, estimates queues, chooses constrained green-time shares, applies robot failure and pointsman fallback capacity, and updates wait time, throughput, queues, and spillback events.

## Model Results

Useful model, MLOps, and benchmark outputs are saved under `model_results/`.

CSV/JSON outputs:

- `best_model_metadata.json`: model features, best Optuna parameters, and test metrics;
- `demo_policy_metrics.csv`: fixed vs adaptive vs failure-aware benchmark metrics;
- `demo_simulation_results.csv`: detailed simulation records;
- `feature_importance.csv`: top model drivers;
- `intersection_config.csv`: simplified Gauteng-like network definition;
- `model_metrics.csv` and `model_metrics.xlsx`: train/test MAE, RMSE, and R2 table;
- `optuna_trials.csv`: Optuna search history;
- `psi_train_vs_test.csv`: train/test drift scores;
- `train_predictions.csv` and `test_predictions.csv`: actual vs predicted demand.

Plots are saved under `model_results/plots/`:

- `psi_train_vs_test.png`: train/test drift using PSI;
- `feature_importance.png`: top model drivers;
- `train_test_prediction_comparison.png`: actual and predicted demand comparison;
- `test_residuals_by_hour.png`: model residuals by hour;
- `optuna_validation_rmse.png`: tuning history;
- `model_test_metrics.png`: selected model MAE, RMSE, and R2;
- `policy_metric_comparison.png`: fixed vs adaptive vs failure-aware metrics;
- `queue_timeseries_by_policy.png`: network queues through the day;
- `intersection_queue_heatmap.png`: queue pressure by intersection;
- `robot_failure_impact.png`: queue impact during robot failures.

## Sources Used For Context

- TomTom Traffic Index, Johannesburg and Pretoria 2025 congestion reports.
- RTMC public mandate: road safety, traffic information, and traffic management coordination.
- Public reporting on South African minibus taxi dependence and road-safety challenges.

## Limitations

- The data is synthetic and designed for a working demonstration, not municipal deployment.
- The network is a simplified four-intersection Gauteng-style corridor, not a full road graph.
- Robot failures are modelled as operational disruptions, not as crime prediction.
- Taxi behaviour is represented as a probabilistic capacity/conflict factor, not individual driver simulation.
