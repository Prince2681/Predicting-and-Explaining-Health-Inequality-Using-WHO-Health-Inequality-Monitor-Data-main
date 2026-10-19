"""
Configuration constants and paths for the WHO Health Inequality project.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"

# File Paths
RAW_DATA_FILE_XLS = RAW_DATA_DIR / "inequality_gap_sex.xls"
RAW_DATA_FILE_CSV = RAW_DATA_DIR / "inequality_gap_sex.csv"
ROOT_DATA_FILE_XLS = BASE_DIR / "inequality_gap_sex.xls"
ROOT_DATA_FILE_CSV = BASE_DIR / "inequality_gap_sex.csv"

SQLITE_DB_PATH = BASE_DIR / "who_capstone.db"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_EXPERIMENT_NAME = "WHO_Inequality_Risk_Classification"

DEFAULT_MODEL_PATH = MODELS_DIR / "module6_random_forest.pkl"

# Modeling Hyperparameters & Columns
FEATURE_COLS = ["date", "region_encoded", "income_encoded", "indicator_scale", "favourable_indicator"]
TARGET_COL_REG = "inequality_gap"
TARGET_COL_CLS = "risk_encoded"

LOW_RISK_CUTOFF = 0.90
MED_RISK_CUTOFF = 4.20
CUTOFF_YEAR = 2020

INCOME_ORDER = [
    "Low-income",
    "Lower-middle-income",
    "Upper-middle-income",
    "High-income"
]

INCOME_MAPPING = {name: idx for idx, name in enumerate(INCOME_ORDER)}

WHO_REGIONS = [
    "African",
    "Americas",
    "Eastern Mediterranean",
    "European",
    "South-East Asia",
    "Western Pacific"
]

REGION_MAPPING = {name: idx for idx, name in enumerate(WHO_REGIONS)}

RISK_TIERS = ["Low", "Medium", "High"]
RISK_MAPPING = {name: idx for idx, name in enumerate(RISK_TIERS)}

# Scale definitions reconstructed from WHO SDG Metadata & Notebook Audits
SCALE_2_INDICATORS = {
    "SE_ALP_CPLR_LOWSEC", "SE_ALP_CPLR_PRIMAR", "SE_ALP_CPLR_UPPSEC",
    "SE_AWP_CPRA_LOWSEC", "SE_AWP_CPRA_PRIMAR", "SE_AWP_CPRA_UPPSEC",
    "SG_DMK_PSRVC", "SG_DMK_PSRVC_ADMINPRO", "SG_DMK_PSRVC_ASSOCPRO",
    "SG_DMK_PSRVC_GENERAL", "SG_DMK_PSRVC_OTHERMAN", "SG_DMK_PSRVC_SENIORMAN"
}

SCALE_1000_INDICATORS = {
    "SH_DYN_IMRT", "SH_DYN_MORT", "SH_HIV_INCD", "PD_AGR_SSFP"
}

SCALE_10000_INDICATORS = {
    "PD_AGR_LSFP", "SI_AGR_SSFP"
}

SCALE_100000_INDICATORS = {
    "SH_STA_POISN", "SH_STA_SCIDE", "VC_IHR_PSRC", "VC_HTF_DETVR",
    "SL_EMP_FTLINJUR", "SL_EMP_INJUR", "SI_AGR_LSFP"
}
