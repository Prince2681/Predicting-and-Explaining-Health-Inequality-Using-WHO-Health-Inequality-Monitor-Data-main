"""
Data loading, cleaning, feature engineering, and preparation module.
Reconstructs and unifies data pipelines from Modules 1-3, 5, and 6.
"""
import os
import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    BASE_DIR,
    RAW_DATA_FILE_XLS,
    RAW_DATA_FILE_CSV,
    ROOT_DATA_FILE_XLS,
    ROOT_DATA_FILE_CSV,
    PROCESSED_DATA_DIR,
    SQLITE_DB_PATH,
    FEATURE_COLS,
    LOW_RISK_CUTOFF,
    MED_RISK_CUTOFF,
    CUTOFF_YEAR,
    INCOME_MAPPING,
    REGION_MAPPING,
    RISK_MAPPING,
    SCALE_2_INDICATORS,
    SCALE_1000_INDICATORS,
    SCALE_10000_INDICATORS,
    SCALE_100000_INDICATORS
)


def load_iso3_region_map() -> dict:
    """Loads pre-extracted ISO3 to WHO region mapping."""
    map_path = BASE_DIR / "iso3_whoreg6_map.json"
    if map_path.exists():
        with open(map_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_indicator_scale(indicator_abbr: str) -> int:
    """Returns the scale factor for a given indicator."""
    if indicator_abbr in SCALE_2_INDICATORS:
        return 2
    if indicator_abbr in SCALE_1000_INDICATORS:
        return 1000
    if indicator_abbr in SCALE_10000_INDICATORS:
        return 10000
    if indicator_abbr in SCALE_100000_INDICATORS:
        return 100000
    return 100


def find_raw_data_file() -> Path:
    """Finds the raw dataset across expected file locations."""
    candidates = [
        RAW_DATA_FILE_CSV,
        RAW_DATA_FILE_XLS,
        ROOT_DATA_FILE_CSV,
        ROOT_DATA_FILE_XLS,
        BASE_DIR / "Predicting-and-Explaining-Health-Inequality-Using-WHO-Health-Inequality-Monitor-Data-main" / "inequality_gap_sex.xls",
        BASE_DIR / "Predicting-and-Explaining-Health-Inequality-Using-WHO-Health-Inequality-Monitor-Data-main" / "inequality_gap_sex.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate inequality_gap_sex dataset file (.xls or .csv).")


def load_raw_dataset(path: Path = None) -> pd.DataFrame:
    """Loads the raw inequality dataset."""
    if path is None:
        path = find_raw_data_file()
    # Note: inequality_gap_sex.xls is formatted as CSV text
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_excel(path)
    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset, maps WHO region and indicator scale,
    encodes ordinal/nominal features, and discretizes the target risk tier.
    """
    model_df = df.copy()

    # 1. Map WHO 6 Regions (whoreg6)
    if "whoreg6" not in model_df.columns:
        iso3_map = load_iso3_region_map()
        model_df["whoreg6"] = model_df["iso3"].map(iso3_map)

    # 2. Map Indicator Scale (indicator_scale)
    if "indicator_scale" not in model_df.columns:
        model_df["indicator_scale"] = model_df["indicator_abbr"].map(get_indicator_scale)

    # 3. Handle missing values: drop records missing region or income
    model_df = model_df.dropna(subset=["whoreg6", "wbincome2025"]).copy()

    # 4. Feature Encoding
    model_df["income_encoded"] = model_df["wbincome2025"].map(INCOME_MAPPING)
    model_df["region_encoded"] = model_df["whoreg6"].map(REGION_MAPPING)

    # 5. Target Formulation for Classification: Risk Tiers
    model_df["abs_gap"] = model_df["inequality_gap"].abs()
    model_df["risk_category"] = pd.cut(
        model_df["abs_gap"],
        bins=[-0.0001, LOW_RISK_CUTOFF, MED_RISK_CUTOFF, float("inf")],
        labels=["Low", "Medium", "High"]
    )
    model_df["risk_encoded"] = model_df["risk_category"].map(RISK_MAPPING).astype(int)

    # Convert data types
    model_df["date"] = model_df["date"].astype(int)
    model_df["indicator_scale"] = model_df["indicator_scale"].astype(int)
    model_df["favourable_indicator"] = model_df["favourable_indicator"].astype(int)

    return model_df


def prepare_train_test_data(
    model_df: pd.DataFrame, cutoff_year: int = CUTOFF_YEAR
):
    """
    Creates time-based train/test splits for both classification and regression.
    Pre-cutoff: Train set
    Post-cutoff: Test set
    """
    X = model_df[FEATURE_COLS].copy()
    y_cls = model_df["risk_encoded"].copy()
    y_reg = model_df["inequality_gap"].copy()

    train_mask = X["date"] < cutoff_year
    test_mask = ~train_mask

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train_cls = y_cls[train_mask]
    y_test_cls = y_cls[test_mask]

    y_train_reg = y_reg[train_mask]
    y_test_reg = y_reg[test_mask]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_cls": y_train_cls,
        "y_test_cls": y_test_cls,
        "y_train_reg": y_train_reg,
        "y_test_reg": y_test_reg,
        "full_data": model_df
    }


def save_processed_datasets(split_dict: dict, output_dir: Path = PROCESSED_DATA_DIR):
    """Saves processed train and test datasets for reusability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dict["X_train"].to_csv(output_dir / "X_train.csv", index=False)
    split_dict["X_test"].to_csv(output_dir / "X_test.csv", index=False)
    split_dict["y_train_cls"].to_csv(output_dir / "y_train.csv", index=False)
    split_dict["y_test_cls"].to_csv(output_dir / "y_test.csv", index=False)
    split_dict["full_data"].to_csv(output_dir / "who_data_cleaned.csv", index=False)


def setup_sqlite_database(df: pd.DataFrame, db_path: Path = SQLITE_DB_PATH):
    """Loads health data into SQLite database and creates index (Module 3)."""
    conn = sqlite3.connect(db_path)
    df.to_sql("health_data", conn, index=False, if_exists="replace")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_health_lookup
        ON health_data(setting, date, indicator_abbr)
    """)
    conn.commit()
    conn.close()
