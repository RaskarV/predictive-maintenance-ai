import shap
import pandas as pd


# --------------------------------------------------
# Create SHAP explainer
# --------------------------------------------------

def create_explainer(xgb_model):
    """
    Create a SHAP TreeExplainer for the trained
    XGBoost model.
    """

    return shap.TreeExplainer(xgb_model)


# --------------------------------------------------
# Get SHAP values for one machine
# --------------------------------------------------

def get_shap_values(explainer, X):
    """
    Calculate SHAP values for the supplied machine data.
    """

    shap_values = explainer.shap_values(X)

    return shap_values


# --------------------------------------------------
# Get important contributors
# --------------------------------------------------

def get_top_contributors(
    shap_values,
    X,
    top_n=5
):
    """
    Return the features that have the largest
    absolute SHAP impact.

    Positive SHAP value:
        pushes prediction toward failure.

    Negative SHAP value:
        pushes prediction away from failure.
    """

    # Handle the single-row case
    if hasattr(shap_values, "ndim") and shap_values.ndim > 1:
        values = shap_values[0]
    else:
        values = shap_values

    feature_values = X.iloc[0]

    results = []

    for feature, shap_value in zip(
        X.columns,
        values
    ):

        results.append({
            "feature": feature,
            "feature_value": float(feature_values[feature]),
            "shap_value": float(shap_value),
            "impact": (
                "Increases failure risk"
                if shap_value > 0
                else "Decreases failure risk"
            )
        })

    # Sort by absolute SHAP impact
    results = sorted(
        results,
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    return pd.DataFrame(
        results[:top_n]
    )