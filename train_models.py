# ============================================
# TRAIN AND SAVE FINAL MODELS
# ============================================

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from src.config import (
    DATA_PATH,
    MODEL_DIR,
    BASE_FEATURES,
    XGB_PARAMS,
    ISOLATION_CONTAMINATION
)

from src.feature_engineering import create_engineered_features


# --------------------------------------------
# 1. Load dataset
# --------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)


# --------------------------------------------
# 2. Prepare original features and target
# --------------------------------------------

X_original = df[BASE_FEATURES].copy()

y = df["Machine failure"]


# --------------------------------------------
# 3. Create engineered features
# --------------------------------------------

X_engineered = create_engineered_features(df)

print(
    "Engineered feature shape:",
    X_engineered.shape
)


# --------------------------------------------
# 4. Train-test split
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_engineered,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------
# 5. Train final XGBoost
# --------------------------------------------

xgb_model = XGBClassifier(
    **XGB_PARAMS
)

xgb_model.fit(
    X_train,
    y_train
)

print("XGBoost trained.")


# --------------------------------------------
# 6. Train Isolation Forest
# --------------------------------------------

# Isolation Forest uses the original five
# sensor features.

X_train_original = X_train[
    [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear"
    ]
]

X_train_normal = X_train_original[
    y_train == 0
]

isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=ISOLATION_CONTAMINATION,
    random_state=42
)

isolation_forest.fit(
    X_train_normal
)

print("Isolation Forest trained.")


# --------------------------------------------
# 7. Create models directory
# --------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------
# 8. Save models
# --------------------------------------------

joblib.dump(
    xgb_model,
    MODEL_DIR / "xgb_engineered.joblib"
)

joblib.dump(
    isolation_forest,
    MODEL_DIR / "isolation_forest.joblib"
)

print("\nModels saved successfully!")
print("Location:", MODEL_DIR)