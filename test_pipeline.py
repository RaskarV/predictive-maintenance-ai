from src.pipeline import analyze_machine


sensor_data = {
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 100
}


result = analyze_machine(sensor_data)


print("\n" + "=" * 50)
print("PREDICTIVE MAINTENANCE RESULT")
print("=" * 50)

print(
    f"Failure Probability : "
    f"{result['failure_probability']:.2%}"
)

print(
    f"Failure Prediction  : "
    f"{result['failure_prediction']}"
)

print(
    f"Anomaly             : "
    f"{'Yes' if result['anomaly_flag'] else 'No'}"
)

print(
    f"Anomaly Score       : "
    f"{result['anomaly_score']:.4f}"
)

print(
    f"Maintenance Status  : "
    f"{result['maintenance_status']}"
)

print("\n" + "-" * 50)
print("TOP SHAP CONTRIBUTORS")
print("-" * 50)

print(
    result["top_contributors"].to_string(index=False)
)

# print("\n" + "-" * 50)
# print("AI MAINTENANCE EXPLANATION")
# print("-" * 50)

print("--------------------------------------------------")
print("AI MAINTENANCE EXPLANATION")
print("--------------------------------------------------")
print(repr(result["explanation"]))

print("=" * 50)