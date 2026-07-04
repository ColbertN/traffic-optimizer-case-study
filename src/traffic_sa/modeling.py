from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import TARGET, build_feature_list, psi_table, time_based_split


def make_one_hot_encoder() -> OneHotEncoder:
    # Support both newer and older scikit-learn versions.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model,
) -> Pipeline:
    # Numeric fields get median imputation and scaling.
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical fields get imputation and one-hot encoding.
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    # ColumnTransformer keeps preprocessing and model training in one reusable pipeline.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_with_optuna(
    df: pd.DataFrame,
    model_path: Path,
    metadata_path: Path,
    trials_path: Path,
    n_trials: int = 25,
    random_state: int = 42,
) -> dict[str, Any]:
    # Reduce Optuna logging so the script output stays readable.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Use a time split because the model must predict future traffic, not shuffled traffic.
    train_df, valid_df, test_df = time_based_split(df)

    # Sample for tuning speed, then retrain the final winner on all train+validation data.
    tune_train_df = train_df.sample(n=min(len(train_df), 15000), random_state=random_state)
    tune_valid_df = valid_df.sample(n=min(len(valid_df), 6000), random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        # Let Optuna decide which feature groups to include.
        use_weather = trial.suggest_categorical("use_weather", [True, False])
        use_reliability = trial.suggest_categorical("use_reliability", [True, False])
        use_lag_features = trial.suggest_categorical("use_lag_features", [True, False])
        use_robot_status = trial.suggest_categorical("use_robot_status", [True, False])
        numeric_features, categorical_features = build_feature_list(
            use_weather=use_weather,
            use_reliability=use_reliability,
            use_lag_features=use_lag_features,
            use_robot_status=use_robot_status,
        )

        # Let Optuna choose the candidate model family and its hyperparameters.
        model_name = trial.suggest_categorical(
            "model_name", ["random_forest", "extra_trees", "gradient_boosting"]
        )
        model = _model_from_trial(trial, model_name, random_state)
        pipeline = build_pipeline(numeric_features, categorical_features, model)

        # Validation RMSE is the optimisation target because the target is vehicle count.
        pipeline.fit(tune_train_df, tune_train_df[TARGET])
        preds = pipeline.predict(tune_valid_df)
        return float(np.sqrt(mean_squared_error(tune_valid_df[TARGET], preds)))

    # TPE is a practical Bayesian search method for small hyperparameter budgets.
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Pull the best feature set, model type, and hyperparameters from Optuna.
    best_params = study.best_trial.params
    best_model_name = best_params["model_name"]

    # Rename the artifact so the filename shows the selected model family.
    model_path = model_path.with_name(
        f"best_{best_model_name}_traffic_demand_pipeline.joblib"
    )
    numeric_features, categorical_features = build_feature_list(
        use_weather=best_params["use_weather"],
        use_reliability=best_params["use_reliability"],
        use_lag_features=best_params["use_lag_features"],
        use_robot_status=best_params["use_robot_status"],
    )

    # Rebuild the winning model from the best Optuna parameters.
    final_model = _model_from_best_params(best_params, random_state)
    final_pipeline = build_pipeline(numeric_features, categorical_features, final_model)

    # Fit the final model on train plus validation before measuring final test performance.
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    final_pipeline.fit(train_valid_df, train_valid_df[TARGET])

    # Generate predictions for model evaluation and later monitoring outputs.
    train_pred = final_pipeline.predict(train_valid_df)
    test_pred = final_pipeline.predict(test_df)

    # Store both train+validation and held-out test metrics.
    metrics = {
        "train_valid": regression_metrics(train_valid_df[TARGET], train_pred),
        "test": regression_metrics(test_df[TARGET], test_pred),
    }

    # Save one artifact containing preprocessing, model, features, target, and metrics.
    artifact = {
        "pipeline": final_pipeline,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET,
        "best_model_name": best_model_name,
        "best_params": best_params,
        "metrics": metrics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)

    # Remove older best-model files so the models folder only keeps the current winner.
    _remove_stale_model_artifacts(model_path)

    # Save every Optuna trial for transparency in the presentation.
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_path, index=False)

    # Metadata is lightweight and easy to inspect without loading the model artifact.
    metadata = {
        "model_path": str(model_path),
        "target": TARGET,
        "n_trials": n_trials,
        "best_model_name": best_model_name,
        "best_trial_number": study.best_trial.number,
        "best_validation_rmse": study.best_value,
        "best_params": best_params,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics": metrics,
        "rows": {
            "train": len(train_df),
            "validation": len(valid_df),
            "test": len(test_df),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Attach predictions to train/test rows for drift and residual monitoring plots.
    train_monitor = train_valid_df.copy()
    test_monitor = test_df.copy()
    train_monitor["prediction"] = train_pred
    test_monitor["prediction"] = test_pred
    return {
        "artifact": artifact,
        "metadata": metadata,
        "train_monitor": train_monitor,
        "test_monitor": test_monitor,
        "psi": psi_table(train_monitor, test_monitor, numeric_features + ["prediction"]),
        "trials": trials_df,
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    # These three metrics give both error size and fit quality.
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def load_model_artifact(model_path: Path) -> dict[str, Any]:
    # Load the saved preprocessing/model bundle for demo simulation.
    return joblib.load(model_path)


def _remove_stale_model_artifacts(current_model_path: Path) -> None:
    # Keep only the current best model file to avoid confusion during presentation.
    patterns = [
        "best_traffic_demand_pipeline.joblib",
        "best_*_traffic_demand_pipeline.joblib",
    ]
    for pattern in patterns:
        for candidate in current_model_path.parent.glob(pattern):
            if candidate.resolve() != current_model_path.resolve():
                try:
                    candidate.unlink(missing_ok=True)
                except PermissionError:
                    # OneDrive/Windows can briefly lock large artifacts. Do not fail training.
                    pass


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    # Get readable feature names after one-hot encoding.
    preprocessor = pipeline.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    return [name.replace("numeric__", "").replace("categorical__", "") for name in names]


def feature_importance_frame(artifact: dict[str, Any]) -> pd.DataFrame:
    # Tree-based models expose feature_importances_, which is useful for explainability.
    pipeline = artifact["pipeline"]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame(
            {
                "feature": transformed_feature_names(pipeline),
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _model_from_trial(trial: optuna.Trial, model_name: str, random_state: int):
    # Build the trial-specific candidate model and hyperparameter search space.
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("rf_n_estimators", 35, 85, step=10),
            max_depth=trial.suggest_int("rf_max_depth", 5, 12),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 5, 28),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", 0.7, 1.0]),
            random_state=random_state,
            n_jobs=1,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=trial.suggest_int("et_n_estimators", 35, 85, step=10),
            max_depth=trial.suggest_int("et_max_depth", 5, 12),
            min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 5, 28),
            max_features=trial.suggest_categorical("et_max_features", ["sqrt", 0.7, 1.0]),
            random_state=random_state,
            n_jobs=1,
        )
    return GradientBoostingRegressor(
        n_estimators=trial.suggest_int("gb_n_estimators", 45, 145, step=25),
        learning_rate=trial.suggest_float("gb_learning_rate", 0.02, 0.12, log=True),
        max_depth=trial.suggest_int("gb_max_depth", 2, 5),
        min_samples_leaf=trial.suggest_int("gb_min_samples_leaf", 4, 24),
        subsample=trial.suggest_float("gb_subsample", 0.72, 1.0),
        random_state=random_state,
    )


def _model_from_best_params(params: dict[str, Any], random_state: int):
    # Recreate the winning model exactly from the best Optuna parameters.
    model_name = params["model_name"]
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_leaf=params["rf_min_samples_leaf"],
            max_features=params["rf_max_features"],
            random_state=random_state,
            n_jobs=1,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=params["et_n_estimators"],
            max_depth=params["et_max_depth"],
            min_samples_leaf=params["et_min_samples_leaf"],
            max_features=params["et_max_features"],
            random_state=random_state,
            n_jobs=1,
        )
    return GradientBoostingRegressor(
        n_estimators=params["gb_n_estimators"],
        learning_rate=params["gb_learning_rate"],
        max_depth=params["gb_max_depth"],
        min_samples_leaf=params["gb_min_samples_leaf"],
        subsample=params["gb_subsample"],
        random_state=random_state,
    )
