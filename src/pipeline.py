import pandas as pd

from src.prediction import load_models, predict_machine
from src.explainability import (
    create_explainer,
    get_shap_values,
    get_top_contributors
)
from src.llm_explanation import generate_maintenance_explanation


# --------------------------------------------------
# Load models and SHAP explainer once
# --------------------------------------------------

xgb_model, isolation_forest = load_models()

explainer = create_explainer(xgb_model)


# --------------------------------------------------
# Complete machine analysis
# --------------------------------------------------

def analyze_machine(sensor_data):
    """
    Run the complete predictive-maintenance pipeline.

    Pipeline:
        Sensor Data
            ↓
        XGBoost Prediction
            ↓
        Isolation Forest
            ↓
        SHAP Explainability
            ↓
        Groq Explanation

    Parameters
    ----------
    sensor_data : dict
        Sensor readings for one machine.

    Returns
    -------
    dict
        Complete machine analysis.
    """

    # ----------------------------------------------
    # 1. ML prediction
    # ----------------------------------------------

    result = predict_machine(
        sensor_data,
        xgb_model,
        isolation_forest
    )

    # ----------------------------------------------
    # 2. SHAP explanation
    # ----------------------------------------------

    shap_values = get_shap_values(
        explainer,
        result["engineered_features"]
    )

    top_contributors = get_top_contributors(
        shap_values,
        result["engineered_features"],
        top_n=5
    )

    # ----------------------------------------------
    # 3. LLM explanation
    # ----------------------------------------------

    explanation = generate_maintenance_explanation(
        failure_probability=result["failure_probability"],
        anomaly_flag=result["anomaly_flag"],
        anomaly_score=result["anomaly_score"],
        maintenance_status=result["maintenance_status"],
        top_contributors=top_contributors
    )

    # ----------------------------------------------
    # 4. Return complete result
    # ----------------------------------------------

    return {
        "failure_probability": result["failure_probability"],
        "failure_prediction": result["failure_prediction"],
        "anomaly_flag": result["anomaly_flag"],
        "anomaly_score": result["anomaly_score"],
        "maintenance_status": result["maintenance_status"],
        "engineered_features": result["engineered_features"],
        "top_contributors": top_contributors,
        "explanation": explanation
    }