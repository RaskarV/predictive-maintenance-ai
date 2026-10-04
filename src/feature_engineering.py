# ============================================
# FEATURE ENGINEERING
# ============================================

import pandas as pd

from .config import BASE_FEATURES, MODEL_FEATURES


def create_engineered_features(data):
    """
    Create the five engineered features used by the final XGBoost model.
    """

    # Keep only the required original sensor features
    X = data[BASE_FEATURES].copy()

    # 1. Temperature difference
    X["temperature_difference"] = (
        X["Process temperature [K]"]
        - X["Air temperature [K]"]
    )

    # 2. Mechanical power proxy
    X["mechanical_power_proxy"] = (
        X["Torque [Nm]"]
        * X["Rotational speed [rpm]"]
    )

    # 3. Temperature ratio
    X["temperature_ratio"] = (
        X["Process temperature [K]"]
        / X["Air temperature [K]"]
    )

    # 4. Torque-speed ratio
    X["torque_speed_ratio"] = (
        X["Torque [Nm]"]
        / X["Rotational speed [rpm]"]
    )

    # 5. Wear-temperature interaction
    X["wear_temperature_interaction"] = (
        X["Tool wear [min]"]
        * X["temperature_difference"]
    )

    # Rename columns for XGBoost
    X.columns = MODEL_FEATURES

    return X