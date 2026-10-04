import joblib
import pandas as pd

from src.config import (
    XGB_MODEL_PATH,
    ISOLATION_FOREST_PATH,
    FAILURE_THRESHOLD
)

from src.feature_engineering import create_engineered_features


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

def load_models():
    """
    Load the trained XGBoost and Isolation Forest models.
    """

    xgb_model = joblib.load(XGB_MODEL_PATH)
    isolation_forest = joblib.load(ISOLATION_FOREST_PATH)

    return xgb_model, isolation_forest


# --------------------------------------------------
# Predict machine condition
# --------------------------------------------------

def predict_machine(sensor_data, xgb_model, isolation_forest):
    """
    Generate failure prediction and anomaly detection
    for a single machine.

    Parameters
    ----------
    sensor_data : dict
        Sensor readings for one machine.

    xgb_model : trained XGBoost model

    isolation_forest : trained Isolation Forest model

    Returns
    -------
    dict
        Prediction results.
    """

    # ----------------------------------------------
    # 1. Convert sensor input into DataFrame
    # ----------------------------------------------

    machine_df = pd.DataFrame([sensor_data])

    # ----------------------------------------------
    # 2. Create engineered features
    # ----------------------------------------------

    X_engineered = create_engineered_features(machine_df)

    # ----------------------------------------------
    # 3. XGBoost failure prediction
    # ----------------------------------------------

    failure_probability = xgb_model.predict_proba(
        X_engineered
    )[0, 1]

    # Apply our selected threshold: 0.60
    failure_prediction = int(
        failure_probability >= FAILURE_THRESHOLD
    )

    # ----------------------------------------------
    # 4. Isolation Forest anomaly detection
    # ----------------------------------------------

    # Isolation Forest was trained using
    # the five original sensor features.
    anomaly_features = X_engineered[
        [
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear"
        ]
    ]

    anomaly_prediction = isolation_forest.predict(
        anomaly_features
    )[0]

    anomaly_score = isolation_forest.decision_function(
        anomaly_features
    )[0]

    # Isolation Forest:
    #  1  = normal
    # -1  = anomaly

    anomaly_flag = int(
        anomaly_prediction == -1
    )

    # ----------------------------------------------
    # 5. Determine maintenance status
    # ----------------------------------------------

    if failure_prediction == 1 and anomaly_flag == 1:

        maintenance_status = "High Concern"

    elif failure_prediction == 1:

        maintenance_status = "Predicted Failure"

    elif anomaly_flag == 1:

        maintenance_status = "Unusual Behavior"

    else:

        maintenance_status = "Normal"

    # ----------------------------------------------
    # 6. Return structured result
    # ----------------------------------------------

    return {
        "failure_probability": float(failure_probability),
        "failure_prediction": failure_prediction,
        "anomaly_flag": anomaly_flag,
        "anomaly_score": float(anomaly_score),
        "maintenance_status": maintenance_status,
        "engineered_features": X_engineered
    }