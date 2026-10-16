"""
Main entrypoint for Predicting and Explaining Health Inequality
Using WHO Health Inequality Monitor Data.

Supports running the full machine learning lifecycle, individual pipeline stages,
hyperparameter optimization, SHAP explainability, MLflow tracking, and single-sample inference.
"""
import argparse
import sys
import os
import subprocess
from pathlib import Path
import pandas as pd

from src.config import (
    BASE_DIR,
    DEFAULT_MODEL_PATH,
    IMAGES_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_COLS,
    WHO_REGIONS,
    INCOME_ORDER,
    RISK_TIERS
)
from src.data_loader import (
    load_raw_dataset,
    clean_and_engineer_features,
    prepare_train_test_data,
    save_processed_datasets,
    setup_sqlite_database
)
from src.eda import (
    compute_gap_statistics,
    perform_statistical_tests,
    generate_eda_plots
)
from src.models import (
    train_regression_models,
    train_classification_models,
    plot_and_save_confusion_matrix,
    tune_random_forest_optuna,
    save_model,
    load_model,
    predict_risk_tier
)
from src.explainability import (
    compute_shap_explanations,
    plot_and_save_importance_comparison,
    explain_sample_prediction
)
from src.tracking import (
    log_all_classification_experiments,
    get_experiment_summary
)


def print_banner():
    banner = r"""
========================================================================================
   WHO HEALTH INEQUALITY MONITOR - PREDICTING AND EXPLAINING HEALTH INEQUALITY
========================================================================================
    """
    print(banner)


def run_pipeline(
    run_eda: bool = True,
    run_reg: bool = True,
    run_cls: bool = True,
    run_tuning: bool = False,
    run_explain: bool = True,
    run_mlflow: bool = True
):
    print("\n[Step 1/6] Loading and Preprocessing WHO Dataset...")
    raw_df = load_raw_dataset()
    print(f" -> Raw data loaded: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns")

    model_df = clean_and_engineer_features(raw_df)
    print(f" -> Feature engineering complete: {model_df.shape[0]:,} valid records")

    splits = prepare_train_test_data(model_df)
    save_processed_datasets(splits)
    print(f" -> Train set (pre-2020): {splits['X_train'].shape[0]:,} rows")
    print(f" -> Test set (2020-2024): {splits['X_test'].shape[0]:,} rows")

    # Set up SQLite database
    setup_sqlite_database(model_df)
    print(" -> SQLite database populated (who_capstone.db)")

    # Step 2: EDA & Statistical Tests
    if run_eda:
        print("\n[Step 2/6] Running Exploratory Data Analysis & Statistical Tests...")
        stats_dict = compute_gap_statistics(model_df)
        print(f" -> Target Skewness: Raw = {stats_dict['raw_skewness']:.2f}, Trimmed 98% = {stats_dict['trimmed_skewness']:.2f}")
        print(f" -> Gap Direction: Negative={stats_dict['negative_pct']:.1f}%, Positive={stats_dict['positive_pct']:.1f}%, Zero={stats_dict['zero_pct']:.1f}%")

        tests = perform_statistical_tests(model_df)
        sw = tests.get("shapiro_wilk", {})
        print(f" -> Shapiro-Wilk Test: stat={sw.get('statistic', 0):.4f}, p={sw.get('p_value', 0):.4e} (Normal: {sw.get('is_normal')})")
        mwu = tests.get("mann_whitney_u", {})
        print(f" -> Mann-Whitney U (High vs Low Income): p={mwu.get('p_value', 0):.4e} (Significant: {mwu.get('is_significant')})")

        plots = generate_eda_plots(model_df)
        print(f" -> EDA Plots saved to: {IMAGES_DIR}")

    # Step 3: Regression Modeling (Module 5)
    if run_reg:
        print("\n[Step 3/6] Training Regression Models (Continuous Inequality Gap)...")
        reg_out = train_regression_models(
            splits["X_train"], splits["y_train_reg"],
            splits["X_test"], splits["y_test_reg"]
        )
        print("\n--- Regression Benchmark Results ---")
        print(reg_out["results"].to_string(index=False))

    # Step 4: Classification Modeling (Module 6)
    cls_models = {}
    if run_cls:
        print("\n[Step 4/6] Training Classification Models (Inequality Risk Tiers: Low/Medium/High)...")
        cls_out = train_classification_models(
            splits["X_train"], splits["y_train_cls"],
            splits["X_test"], splits["y_test_cls"]
        )
        cls_models = cls_out["models"]
        print("\n--- Classification Benchmark Results ---")
        print(cls_out["results"].to_string(index=False))

        # Best primary model: Random Forest
        rf_model = cls_models.get("Random Forest")
        if rf_model is not None:
            save_model(rf_model, DEFAULT_MODEL_PATH)
            print(f" -> Primary Random Forest model saved to: {DEFAULT_MODEL_PATH}")

            y_pred = rf_model.predict(splits["X_test"])
            cm_path = plot_and_save_confusion_matrix(splits["y_test_cls"], y_pred)
            print(f" -> Confusion matrix saved to: {cm_path}")

        # Optional Optuna tuning
        if run_tuning:
            print("\nRunning Optuna Hyperparameter Optimization (20 trials)...")
            tuning_out = tune_random_forest_optuna(splits["X_train"], splits["y_train_cls"], n_trials=20)
            print(f" -> Best Hyperparameters: {tuning_out['best_params']}")
            print(f" -> Best CV Accuracy: {tuning_out['best_cv_score']*100:.2f}%")
            cls_models["Random Forest (Optuna Tuned)"] = tuning_out["best_model"]

    # Step 5: SHAP Explainability (Module 7)
    if run_explain:
        print("\n[Step 5/6] Computing SHAP Model Explainability...")
        rf_model = cls_models.get("Random Forest")
        if rf_model is None and DEFAULT_MODEL_PATH.exists():
            rf_model = load_model(DEFAULT_MODEL_PATH)

        if rf_model is not None:
            # Sample 500 rows for explainability
            X_sample = splits["X_test"].sample(min(500, len(splits["X_test"])), random_state=42)
            shap_res = compute_shap_explanations(rf_model, X_sample)
            print("\n--- Global Feature Importance (SHAP vs Gini) ---")
            print(shap_res["global_importance"].to_string(index=False))

            shap_plot = plot_and_save_importance_comparison(
                shap_res["rf_importance"],
                shap_res["global_importance"].set_index("Feature")["Mean_Abs_SHAP"]
            )
            print(f" -> Feature importance comparison plot saved to: {shap_plot}")

            # Explain sample prediction
            sample_row = splits["X_test"].iloc[[0]]
            sample_exp = explain_sample_prediction(shap_res["explainer"], rf_model, sample_row)
            print("\n--- Example Individual Sample Explanation ---")
            print(f"Predicted Tier: {sample_exp['predicted_tier']}")
            print(f"Probabilities: {sample_exp['probabilities']}")
            print("SHAP Feature Contributions:")
            for feat, val in sample_exp["contributions"].items():
                print(f"  {feat:<22} {val:+.4f}")

    # Step 6: MLflow Experiment Tracking (Module 7)
    if run_mlflow and cls_models:
        print("\n[Step 6/6] Logging Experiments to MLflow...")
        try:
            summary = log_all_classification_experiments(
                cls_models, splits["X_test"], splits["y_test_cls"]
            )
            print("\n--- MLflow Experiment Runs Logged ---")
            print(summary.to_string(index=False))
            print(" -> MLflow tracking database: mlflow.db")
        except Exception as e:
            print(f"Note: MLflow logging encountered: {e}")

    print("\n========================================================================================")
    print(" Pipeline execution finished successfully!")
    print(f" Artifacts generated in: {BASE_DIR}")
    print("========================================================================================\n")


def handle_prediction(args):
    """Handles single custom prediction from CLI."""
    if not DEFAULT_MODEL_PATH.exists():
        print(f"Model not found at {DEFAULT_MODEL_PATH}. Running training first...")
        run_pipeline(run_eda=False, run_reg=False, run_tuning=False, run_explain=False, run_mlflow=False)

    model = load_model(DEFAULT_MODEL_PATH)
    res = predict_risk_tier(
        model=model,
        date=args.year,
        region=args.region,
        income_group=args.income,
        indicator_scale=args.scale,
        favourable_indicator=args.favourable
    )

    print("\n=======================================================")
    print(" World Health Organization | Health Inequality Monitor")
    print(" Real-Time Inequality Risk Stratification Assessment")
    print("=======================================================")
    print(f"Parameters: Year={args.year}, Region={args.region}, Income={args.income}, Scale={args.scale}, Favourable={args.favourable}")
    print(f"Stratified Inequality Risk: [{res['predicted_tier'].upper()} RISK]")
    print("Posterior Probabilities:")
    for tier, prob in res["probabilities"].items():
        print(f"  {tier:<8}: {prob*100:.1f}%")
    print("=======================================================\n")


def launch_streamlit():
    """Launches interactive Streamlit web dashboard."""
    app_path = BASE_DIR / "app.py"
    print(f"Launching Streamlit application from {app_path} on port 8501...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port=8501", "--server.address=0.0.0.0"]
    subprocess.run(cmd)


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Predicting and Explaining Health Inequality Using WHO Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--all", action="store_true", help="Run full end-to-end pipeline (default if no args)")
    parser.add_argument("--eda", action="store_true", help="Run Exploratory Data Analysis & generate charts")
    parser.add_argument("--train-reg", action="store_true", help="Train regression models (Module 5)")
    parser.add_argument("--train-cls", action="store_true", help="Train classification models (Module 6)")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--explain", action="store_true", help="Run SHAP explainability analysis (Module 7)")
    parser.add_argument("--mlflow", action="store_true", help="Log models and metrics to MLflow")
    parser.add_argument("--ui", action="store_true", help="Launch interactive Streamlit web application")

    # Prediction arguments
    pred_group = parser.add_argument_group("Inference options")
    pred_group.add_argument("--predict", action="store_true", help="Make a single prediction with provided options")
    pred_group.add_argument("--year", type=int, default=2024, help="Year of measurement")
    pred_group.add_argument("--region", type=str, default="European", choices=WHO_REGIONS, help="WHO Region")
    pred_group.add_argument("--income", type=str, default="High-income", choices=INCOME_ORDER, help="World Bank Income Group")
    pred_group.add_argument("--scale", type=int, default=100, choices=[2, 100, 1000, 10000, 100000], help="Indicator Scale")
    pred_group.add_argument("--favourable", type=int, default=0, choices=[0, 1], help="Is indicator favourable (0=unfavourable, 1=favourable)")

    args = parser.parse_args()

    # Route according to arguments
    if args.ui:
        launch_streamlit()
        return

    if args.predict:
        handle_prediction(args)
        return

    # Check if specific stages requested
    specific_requested = any([args.eda, args.train_reg, args.train_cls, args.tune, args.explain, args.mlflow])

    if specific_requested:
        run_pipeline(
            run_eda=args.eda,
            run_reg=args.train_reg,
            run_cls=args.train_cls,
            run_tuning=args.tune,
            run_explain=args.explain,
            run_mlflow=args.mlflow
        )
    else:
        # Default: full end-to-end pipeline
        run_pipeline(
            run_eda=True,
            run_reg=True,
            run_cls=True,
            run_tuning=args.tune,
            run_explain=True,
            run_mlflow=True
        )


if __name__ == "__main__":
    main()
