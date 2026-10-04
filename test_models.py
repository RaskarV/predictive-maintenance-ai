import joblib
import pandas as pd

from src.feature_engineering import create_engineered_features


# --------------------------------------------------
# 1. Load saved models
# --------------------------------------------------

print("Loading saved models...")

xgb_model = joblib.load("models/xgb_engineered.joblib")
isolation_forest = joblib.load("models/isolation_forest.joblib")

print("✅ XGBoost model loaded")
print("✅ Isolation Forest model loaded")


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/raw/ai4i2020.csv")

print(f"✅ Dataset loaded: {df.shape}")


# --------------------------------------------------
# 3. Select one machine
# --------------------------------------------------

machine = df.iloc[[0]].copy()

print("\nSelected machine:")
print(machine[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
])


# --------------------------------------------------
# 4. Create engineered features
# --------------------------------------------------

X_engineered = create_engineered_features(machine)

print("\nEngineered features:")
print(X_engineered)


# --------------------------------------------------
# 5. XGBoost prediction
# --------------------------------------------------

failure_probability = xgb_model.predict_proba(X_engineered)[0, 1]

print("\nXGBoost prediction:")
print(f"Failure probability: {failure_probability:.4f}")


# --------------------------------------------------
# 6. Isolation Forest prediction
# --------------------------------------------------

# Isolation Forest uses the original five sensor features
X_anomaly = X_engineered[
    [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear"
    ]
]

anomaly_prediction = isolation_forest.predict(X_anomaly)[0]
anomaly_score = isolation_forest.decision_function(X_anomaly)[0]

anomaly_flag = 1 if anomaly_prediction == -1 else 0

print("\nIsolation Forest:")
print(f"Anomaly prediction: {anomaly_prediction}")
print(f"Anomaly flag: {anomaly_flag}")
print(f"Anomaly score: {anomaly_score:.4f}")


# --------------------------------------------------
# 7. Final status
# --------------------------------------------------

FAILURE_THRESHOLD = 0.70

if failure_probability >= FAILURE_THRESHOLD and anomaly_flag == 1:
    status = "High Concern"

elif failure_probability >= FAILURE_THRESHOLD:
    status = "Predicted Failure"

elif anomaly_flag == 1:
    status = "Unusual Behavior"

else:
    status = "Normal"


print("\n" + "=" * 50)
print("FINAL MAINTENANCE RESULT")
print("=" * 50)

print(f"Failure Probability : {failure_probability:.2%}")
print(f"Anomaly             : {'Yes' if anomaly_flag else 'No'}")
print(f"Anomaly Score       : {anomaly_score:.4f}")
print(f"Maintenance Status  : {status}")
print("=" * 50)