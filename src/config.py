# ============================================
# PROJECT CONFIGURATION
# ============================================

from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_PATH = BASE_DIR / "data" / "raw" / "ai4i2020.csv"

# Models
MODEL_DIR = BASE_DIR / "models"

XGB_MODEL_PATH = MODEL_DIR / "xgb_engineered.joblib"
ISOLATION_FOREST_PATH = MODEL_DIR / "isolation_forest.joblib"

# Prediction threshold
FAILURE_THRESHOLD = 0.60

# Isolation Forest configuration
ISOLATION_CONTAMINATION = 0.03

# XGBoost configuration
XGB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "scale_pos_weight": 28.5,
    "random_state": 42,
    "eval_metric": "logloss"
}

# Original sensor features
BASE_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Final model feature names
MODEL_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "temperature_difference",
    "mechanical_power_proxy",
    "temperature_ratio",
    "torque_speed_ratio",
    "wear_temperature_interaction"
]