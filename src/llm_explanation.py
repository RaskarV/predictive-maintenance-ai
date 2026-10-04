import os

from dotenv import load_dotenv
from groq import Groq


# Load environment variables
load_dotenv()


def generate_maintenance_explanation(
    failure_probability,
    anomaly_flag,
    anomaly_score,
    maintenance_status,
    top_contributors
):
    """
    Generate a concise maintenance explanation using Groq.

    Groq is used only to explain the existing ML results.
    It does not make or modify the prediction.
    """

    # --------------------------------------------------
    # API configuration
    # --------------------------------------------------

    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b"
    )

    if not api_key:
        return (
            "Groq explanation is unavailable because "
            "GROQ_API_KEY is not configured."
        )

    client = Groq(api_key=api_key)


    # --------------------------------------------------
    # Convert SHAP results into controlled text
    # --------------------------------------------------

    shap_lines = []

    feature_units = {
        "air_temperature": "K",
        "process_temperature": "K",
        "rotational_speed": "rpm",
        "torque": "Nm",
        "tool_wear": "min",
        "temperature_difference": "K",
        "mechanical_power_proxy": "proxy units",
        "temperature_ratio": "ratio",
        "torque_speed_ratio": "ratio",
        "wear_temperature_interaction": "derived units"
    }

    for _, row in top_contributors.iterrows():

        feature = row["feature"]
        value = row["feature_value"]
        shap_value = row["shap_value"]

        unit = feature_units.get(feature, "")

        if shap_value > 0:
            direction = (
                "positive contribution toward the "
                "model's failure output"
            )
        else:
            direction = (
                "negative contribution toward the "
                "model's failure output"
            )

        shap_lines.append(
            f"- {feature}: value={value:.4f} {unit}, "
            f"SHAP={shap_value:.4f}, {direction}."
        )

    shap_text = "\n".join(shap_lines)


    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = f"""
You are an industrial predictive-maintenance assistant.

Your ONLY job is to explain an existing machine-learning result
to a maintenance engineer.

IMPORTANT RULES:

- Do not make a new prediction.
- Do not change or override the ML prediction.
- Do not invent sensor readings.
- Do not invent engineering specifications or operating ranges.
- Do not invent maintenance history.
- Do not claim that SHAP proves causation.
- Clearly distinguish model prediction from anomaly detection.
- A positive SHAP value means that the feature contributed toward
  increasing the model output for this particular prediction.
- A negative SHAP value means that the feature contributed toward
  decreasing the model output for this particular prediction.
- Do NOT say that a feature physically causes or prevents machine
  failure based only on its SHAP value.
- Do NOT describe a feature as "normal", "typical", "safe",
  "balanced", or "within normal bounds" unless an explicit
  reference range has been provided.
- Use the supplied feature units exactly.
- Tool wear is measured in minutes, NOT percentage.
- mechanical_power_proxy is a derived model feature. It is NOT
  actual measured mechanical power and must NOT be expressed in kW.
- Do not convert derived features into physical units.
- Refer to SHAP values as influences on the model output,
  not physical causes.

ML RESULTS:

Failure probability: {failure_probability:.4f}
Isolation Forest anomaly flag: {anomaly_flag}
Isolation Forest anomaly score: {anomaly_score:.4f}
Maintenance status: {maintenance_status}

TOP SHAP CONTRIBUTORS:

{shap_text}

Provide the response in exactly this structure:

1. Overall assessment
State the failure probability, anomaly result, and maintenance status.
Do not add unsupported engineering claims.

2. Key factors
For each important SHAP contributor, state:
- feature name
- feature value with its supplied unit
- SHAP value
- whether it contributed positively or negatively
  to the model output

Use wording such as:
"Tool wear was 100 min and contributed negatively to the
model output (SHAP = -1.50)."

Do NOT say that the feature itself physically increases
or decreases failure risk.

3. Maintenance recommendation
Give a cautious recommendation based only on the ML prediction,
anomaly result, and SHAP information.

Do not diagnose a specific component.
Do not claim that a sensor value is within an engineering
operating range.

Keep the answer under 150 words.
"""


    # --------------------------------------------------
    # Groq API call
    # --------------------------------------------------

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
         reasoning_effort="low",
        include_reasoning=False,
        max_completion_tokens=1000
    )

    return response.choices[0].message.content