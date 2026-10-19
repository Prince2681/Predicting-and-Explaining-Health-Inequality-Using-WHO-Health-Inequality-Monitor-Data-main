"""
Experiment Tracking Module using MLflow (Module 7).
Logs runs, parameters, metrics, and models to SQLite-backed MLflow tracking store.
"""
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.models import evaluate_classification


def setup_mlflow_tracking(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME
):
    """Initializes MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_model_run(
    model,
    run_name: str,
    params: dict,
    X_test,
    y_test,
    flavor: str = "sklearn"
) -> dict:
    """Logs a single model training run to MLflow."""
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)

    metrics = evaluate_classification(y_test, y_pred, y_prob, run_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", metrics["Accuracy"])
        mlflow.log_metric("macro_f1", metrics["Macro_F1"])
        if not pd.isna(metrics["Macro_ROC_AUC"]):
            mlflow.log_metric("macro_roc_auc", metrics["Macro_ROC_AUC"])

        try:
            if flavor == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    skops_trusted_types=["sklearn.tree._tree.Tree"]
                )
        except Exception as e:
            # Continue even if model artifact logging encounters serialization quirks
            pass

    return metrics


def log_all_classification_experiments(
    fitted_models: dict,
    X_test,
    y_test
) -> pd.DataFrame:
    """Logs all benchmark classification models to MLflow."""
    setup_mlflow_tracking()

    run_configs = [
        ("Baseline", {"strategy": "stratified"}, "sklearn"),
        ("Logistic Regression", {"max_iter": 1000}, "sklearn"),
        ("Decision Tree", {"max_depth": 10}, "sklearn"),
        ("Random Forest", {"n_estimators": 200, "max_depth": 10}, "sklearn"),
        ("XGBoost", {"n_estimators": 200, "max_depth": 6}, "xgboost"),
    ]

    logged_runs = []
    for name, params, flavor in run_configs:
        if name in fitted_models:
            m = fitted_models[name]
            metrics = log_model_run(m, name, params, X_test, y_test, flavor=flavor)
            logged_runs.append(metrics)

    return pd.DataFrame(logged_runs)


def get_experiment_summary(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> pd.DataFrame:
    """Fetches logged runs from MLflow and returns leaderboard."""
    try:
        runs = mlflow.search_runs(experiment_names=[experiment_name])
        if runs.empty:
            return pd.DataFrame()
        display_cols = [c for c in [
            "tags.mlflow.runName", "metrics.accuracy", "metrics.macro_f1", "metrics.macro_roc_auc"
        ] if c in runs.columns]
        return runs[display_cols].sort_values("metrics.accuracy", ascending=False)
    except Exception:
        return pd.DataFrame()
