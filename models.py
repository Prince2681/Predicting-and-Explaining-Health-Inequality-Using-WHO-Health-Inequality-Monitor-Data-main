"""
Machine learning models for Regression (Module 5) and Classification (Module 6).
Includes model training, evaluation, hyperparameter tuning, and inference.
"""
from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb
import optuna

from src.config import (
    MODELS_DIR,
    IMAGES_DIR,
    DEFAULT_MODEL_PATH,
    FEATURE_COLS,
    RISK_TIERS,
    INCOME_MAPPING,
    REGION_MAPPING
)


# ==========================================
# Regression Models (Module 5)
# ==========================================

def evaluate_regression(y_true, y_pred, name: str) -> dict:
    """Calculates regression metrics (MAE, RMSE, R²)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "model": name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


def train_regression_models(X_train, y_train, X_test, y_test) -> dict:
    """Trains regression models to predict continuous inequality_gap."""
    models = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred, name)
        results.append(metrics)
        fitted_models[name] = model

    results_df = pd.DataFrame(results)
    return {
        "results": results_df,
        "models": fitted_models
    }


# ==========================================
# Classification Models (Module 6)
# ==========================================

def evaluate_classification(y_true, y_pred, y_prob=None, name: str = "") -> dict:
    """Calculates classification metrics (Accuracy, Macro F1, Macro ROC-AUC)."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc = None

    return {
        "model": name,
        "Accuracy": float(acc),
        "Macro_F1": float(f1),
        "Macro_ROC_AUC": float(auc) if auc is not None else np.nan
    }


def train_classification_models(X_train, y_train, X_test, y_test) -> dict:
    """Trains baseline, linear, tree, and ensemble classification models."""
    models = {
        "Baseline": DummyClassifier(strategy="stratified", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

        metrics = evaluate_classification(y_test, y_pred, y_prob, name)
        results.append(metrics)
        fitted_models[name] = model

    results_df = pd.DataFrame(results)
    return {
        "results": results_df,
        "models": fitted_models
    }


def plot_and_save_confusion_matrix(
    y_true, y_pred, output_path: Path = IMAGES_DIR / "06_confusion_matrix.png"
):
    """Plots and saves confusion matrix heatmap for classification."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="crest",
        xticklabels=RISK_TIERS,
        yticklabels=RISK_TIERS,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Random Forest\nInequality Risk Classification")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def tune_random_forest_optuna(X_train, y_train, n_trials: int = 15) -> dict:
    """Optimizes Random Forest hyperparameters using Optuna (Module 6)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        # 3-fold cross validation on subsample for efficient execution
        sample_idx = np.random.choice(len(X_train), size=min(10000, len(X_train)), replace=False)
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(model, X_train.iloc[sample_idx], y_train.iloc[sample_idx], cv=3, scoring="accuracy", n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_rf = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train)

    return {
        "best_params": study.best_params,
        "best_cv_score": study.best_value,
        "best_model": best_rf
    }


def save_model(model, filepath: Path = DEFAULT_MODEL_PATH):
    """Serializes model to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path = DEFAULT_MODEL_PATH):
    """Loads serialized model from disk."""
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found at {filepath}")
    return joblib.load(filepath)


def predict_risk_tier(
    model,
    date: int,
    region: str,
    income_group: str,
    indicator_scale: int,
    favourable_indicator: int
) -> dict:
    """Makes a real-time risk tier prediction for single sample."""
    region_encoded = REGION_MAPPING.get(region, 0)
    income_encoded = INCOME_MAPPING.get(income_group, 0)

    sample = pd.DataFrame([{
        "date": int(date),
        "region_encoded": int(region_encoded),
        "income_encoded": int(income_encoded),
        "indicator_scale": int(indicator_scale),
        "favourable_indicator": int(favourable_indicator)
    }])[FEATURE_COLS]

    pred_class_idx = int(model.predict(sample)[0])
    pred_tier = RISK_TIERS[pred_class_idx]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)[0]
        for idx, tier in enumerate(RISK_TIERS):
            probabilities[tier] = float(probs[idx])

    return {
        "predicted_tier": pred_tier,
        "predicted_class_index": pred_class_idx,
        "probabilities": probabilities,
        "sample": sample
    }
