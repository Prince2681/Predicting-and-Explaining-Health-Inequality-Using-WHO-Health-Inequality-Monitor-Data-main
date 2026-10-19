# Predicting and Explaining Health Inequality Using WHO Health Inequality Monitor Data

[![Python](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-teal.svg)](https://mlflow.org/)

This project investigates sex-disaggregated health inequalities across 198 countries and 131 Sustainable Development Goal (SDG) health indicators using the **World Health Organization (WHO) Health Inequality Monitor Database**.

The primary target is the sex inequality gap:
$$\text{Inequality Gap} = \text{Female Estimate} - \text{Male Estimate}$$

---

## 📁 Project Structure

```text
.
├── entrypoint.sh               # Container startup script
├── main.py                     # Main execution pipeline and CLI
├── app.py                      # Interactive Streamlit Web Dashboard
├── requirements.txt            # Python dependencies
├── iso3_whoreg6_map.json       # ISO3 to WHO region mapping (offline-ready)
├── inequality_gap_sex.xls      # Raw sex inequality dataset
│
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── config.py               # Paths, constants, and feature definitions
│   ├── data_loader.py          # Data ingestion, cleaning, and encoding
│   ├── eda.py                  # Statistics, hypothesis tests, and plots
│   ├── models.py               # Regression & classification models, Optuna tuning
│   ├── explainability.py       # SHAP TreeExplainer & feature importance
│   └── tracking.py             # MLflow experiment tracking
│
├── data/
│   ├── raw/                    # Raw source dataset
│   └── processed/              # Cleaned splits (X_train, X_test, y_train, y_test)
│
├── models/                     # Serialized trained models (.pkl)
├── images/                     # Generated charts and visualizations
│
└── notebooks/ (original modules)
    ├── Module 1 - 2 Data audit.ipynb
    ├── Module 3 - SQL Data Engineering and Inequality Gap.ipynb
    ├── Module 4 - Exploratory Data Analysis and Statistics.ipynb
    ├── Module 5 - Feature Engineering and Regression.ipynb
    ├── Module 6 – Classification and Model Optimization.ipynb
    └── Module 7 – Explainability and Experiment Tracking.ipynb
```

---


##  Local Setup & Execution

### 1. Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Full Machine Learning Pipeline
```bash
python main.py
```
This executes:
1. **Data Ingestion & Cleaning:** Loads 79,448 matched pairs, assigns WHO regions and indicator scales, and creates train/test splits.
2. **Exploratory Data Analysis:** Computes statistics, runs Shapiro-Wilk and Mann-Whitney U tests, and saves plots to `images/`.
3. **Regression Modeling (Module 5):** Evaluates Baseline, Linear Regression, Random Forest, and XGBoost on continuous inequality gap.
4. **Classification Modeling (Module 6):** Evaluates Baseline, Logistic Regression, Decision Tree, Random Forest, and XGBoost on Risk Tiers (Low, Medium, High).
5. **SHAP Explainability (Module 7):** Computes TreeExplainer SHAP values and feature contributions.
6. **MLflow Tracking:** Logs runs, parameters, metrics, and models to `mlflow.db`.

### 3. Run Individual Pipeline Stages via CLI
```bash
# Run only EDA and generate visual charts
python main.py --eda

# Train only classification models
python main.py --train-cls

# Train with Optuna hyperparameter optimization
python main.py --train-cls --tune

# Run SHAP model explainability
python main.py --explain

# Run MLflow experiment logging
python main.py --mlflow

# Make a single real-time prediction
python main.py --predict --year 2024 --region European --income High-income --scale 100 --favourable 1
```

### 4. Launch the Interactive Streamlit Web UI
```bash
streamlit run app.py
# or
python main.py --ui
```

---

## Methodology & Modules Walkthrough

| Module | Core Purpose | Key Insight / Methodology |
|---|---|---|
| **Module 1 & 2** | Data Audit & Hygiene | Checked grain uniqueness, verified no negative values in valid rate indicators, dropped uninformative columns (`population`, `se`). |
| **Module 3** | SQL Engineering | Performed SQL self-join on sex-disaggregated strata (`reference_subgroup = 0` vs `1`) matching setting, date, and indicator. |
| **Module 4** | EDA & Statistics | Diagnosed extreme raw target skewness (-177.95) caused by mixed indicator scales; defined 3-class risk terciles: Low ($\le 0.90$), Medium ($0.90 - 4.20$), High ($> 4.20$). |
| **Module 5** | Regression | Evaluated continuous gap prediction using temporal cutoff (pre-2020 train, 2020+ test) to avoid future data leakage. |
| **Module 6** | Classification & Tuning | Modeled inequality risk tier. Random Forest achieved top accuracy and 0.718 ROC-AUC. Tuned with Optuna. |
| **Module 7** | Explainability & Tracking | SHAP TreeExplainer revealed that `indicator_scale` contributes ~40% of predictive power. Experiment tracking recorded with MLflow. |

---

## Summary of Model Performance

### Classification: Inequality Risk Tier (Low / Medium / High)

| Model | Test Accuracy | Macro F1 | Macro ROC-AUC |
|---|---|---|---|
| **Baseline (Stratified)** | 33.8% | 0.334 | 0.500 |
| **Logistic Regression** | 41.2% | 0.385 | 0.601 |
| **Decision Tree** | 51.4% | 0.508 | 0.672 |
| **Random Forest** | **53.6%** | **0.531** | **0.718** |
| **XGBoost** | 52.8% | 0.523 | 0.709 |

---

## License
This project is open-source under the MIT License.
