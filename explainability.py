"""
Explainability Module using SHAP (Module 7).
Computes global feature importance and individual prediction attributions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.config import IMAGES_DIR, FEATURE_COLS, RISK_TIERS


def compute_shap_explanations(model, X_sample: pd.DataFrame) -> dict:
    """Computes TreeExplainer SHAP values and summarizes feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # shap_values shape can be:
    # (n_samples, n_features, n_classes) or list of (n_samples, n_features)
    if isinstance(shap_values, list):
        # List of arrays per class
        shap_array = np.stack(shap_values, axis=-1)
    else:
        shap_array = np.array(shap_values)

    # Mean absolute SHAP across samples and classes
    if shap_array.ndim == 3:
        mean_abs_shap = np.abs(shap_array).mean(axis=(0, 2))
    elif shap_array.ndim == 2:
        mean_abs_shap = np.abs(shap_array).mean(axis=0)
    else:
        mean_abs_shap = np.zeros(len(FEATURE_COLS))

    global_importance_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False)

    rf_importance = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "global_importance": global_importance_df,
        "rf_importance": rf_importance
    }


def plot_and_save_importance_comparison(
    rf_importance: pd.Series,
    shap_importance: pd.Series,
    output_path: Path = IMAGES_DIR / "07_importance_comparison.png"
):
    """Plots side-by-side bar charts comparing Random Forest Gini vs SHAP importances."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].barh(rf_importance.index[::-1], rf_importance.values[::-1], color="#E8584A")
    axes[0].set_title("Random Forest Built-in Importance\n(Gini-based)")
    axes[0].set_xlabel("Gini Importance")

    axes[1].barh(shap_importance.index[::-1], shap_importance.values[::-1], color="#4A90D9")
    axes[1].set_title("SHAP Feature Importance\n(Marginal Contribution-based)")
    axes[1].set_xlabel("Mean |SHAP Value|")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def explain_sample_prediction(explainer, model, sample_row: pd.DataFrame) -> dict:
    """Explains a single prediction with class probabilities and SHAP contributions."""
    proba = model.predict_proba(sample_row)[0]
    pred_idx = int(model.predict(sample_row)[0])
    sv = explainer.shap_values(sample_row)

    if isinstance(sv, list):
        # List per class: sv[class_idx] has shape (1, n_features)
        class_shap = sv[pred_idx][0]
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        class_shap = sv[0, :, pred_idx]
    else:
        class_shap = sv[0]

    contributions = dict(zip(FEATURE_COLS, [float(v) for v in class_shap]))

    return {
        "predicted_tier": RISK_TIERS[pred_idx],
        "probabilities": {RISK_TIERS[i]: float(p) for i, p in enumerate(proba)},
        "contributions": contributions
    }
