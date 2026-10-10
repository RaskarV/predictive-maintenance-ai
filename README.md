# Predictive Maintenance AI

An end-to-end **AI-powered Predictive Maintenance System** that uses machine sensor data to predict equipment failure, detect abnormal behavior, explain model decisions using SHAP, and generate human-readable maintenance insights using an LLM.

---

## Overview

Unexpected machine failures can lead to production downtime, maintenance costs, and operational losses.

This project uses historical machine sensor data to build a predictive maintenance pipeline that:

- Predicts the probability of machine failure using **XGBoost**
- Detects unusual machine behavior using **Isolation Forest**
- Explains predictions using **SHAP**
- Generates concise maintenance explanations using **Groq LLM**
- Combines model outputs into an interpretable **maintenance status**
- Provides a reusable inference pipeline for application integration

### Complete Pipeline

```text
Sensor Data
     ↓
Feature Engineering
     ↓
XGBoost Failure Prediction
     ↓
Isolation Forest Anomaly Detection
     ↓
SHAP Explainability
     ↓
Groq LLM Explanation
     ↓
Maintenance Recommendation
```

---

## Objectives

The main objectives of this project are:

1. Predict the likelihood of machine failure before it occurs.
2. Detect unusual operating behavior using anomaly detection.
3. Identify the features contributing most to the prediction.
4. Convert model outputs into understandable maintenance insights.
5. Build a modular and reusable machine-learning pipeline.

---

## Dataset

The project uses a machine predictive-maintenance dataset containing sensor measurements and a binary machine-failure target.

### Input Features

| Feature | Description |
|---|---|
| Air temperature [K] | Air temperature around the machine |
| Process temperature [K] | Process operating temperature |
| Rotational speed [rpm] | Machine rotational speed |
| Torque [Nm] | Applied machine torque |
| Tool wear [min] | Tool usage/wear duration |

### Target

```text
Machine failure
```

Binary target:

```text
0 → No failure
1 → Failure
```

The dataset contains:

```text
10,000 rows
```

with a highly imbalanced target:

```text
No Failure : 9661
Failure    : 339
```

---

## Feature Engineering

In addition to the original five sensor features, the project creates engineered features to capture relationships between machine parameters.

### Final Model Features

```text
air_temperature
process_temperature
rotational_speed
torque
tool_wear
temperature_difference
mechanical_power_proxy
temperature_ratio
torque_speed_ratio
wear_temperature_interaction
```

### Engineered Features

#### Temperature Difference

```text
process_temperature - air_temperature
```

Captures the temperature gap between the process and surrounding air.

#### Mechanical Power Proxy

```text
torque × rotational_speed
```

Provides a proxy for mechanical power/load.

#### Temperature Ratio

```text
process_temperature / air_temperature
```

Represents the relative temperature relationship.

#### Torque-Speed Ratio

```text
torque / rotational_speed
```

Captures the relationship between applied torque and rotational speed.

#### Wear-Temperature Interaction

```text
tool_wear × air_temperature
```

Captures the interaction between tool wear and operating temperature.

---

# 🤖 Machine Learning Models

## 1. XGBoost

XGBoost is the primary supervised learning model.

It predicts:

```text
Probability of machine failure
```

The probability is converted into a failure prediction using the configured threshold.

Current threshold:

```text
0.60
```

Therefore:

```text
Failure Probability ≥ 60%
        ↓
Failure Predicted

Failure Probability < 60%
        ↓
No Failure Predicted
```

---

## 2. Isolation Forest

Isolation Forest is used for **anomaly detection**.

Unlike XGBoost, it does not directly predict machine failure.

It identifies whether the current machine behavior is unusual compared with normal operating conditions.

Output:

```text
1  → Normal
-1 → Anomaly
```

The project converts this into:

```text
0 → No anomaly
1 → Anomaly
```

It also produces an anomaly score using:

```python
isolation_forest.decision_function()
```

---

# Maintenance Decision Logic

The system combines the XGBoost failure prediction and Isolation Forest anomaly detection.

| Failure Prediction | Anomaly | Maintenance Status |
|---|---|---|
| No | No | **Normal** |
| Yes | No | **Predicted Failure** |
| No | Yes | **Unusual Behavior** |
| Yes | Yes | **High Concern** |

This provides a more informative maintenance decision than relying on a single model.

---

# Explainable AI with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to understand why the XGBoost model produces a particular prediction.

A SHAP value represents how a feature contributes to the model output.

### Positive SHAP

```text
Pushes the model toward higher failure risk
```

### Negative SHAP

```text
Pushes the model toward lower failure risk
```

The system ranks features according to their absolute SHAP impact and returns the top contributors.

### Example

For one test machine, the top contributors were:

```text
tool_wear
mechanical_power_proxy
rotational_speed
torque_speed_ratio
temperature_ratio
```

All had negative SHAP values for that particular prediction, meaning they pushed the model toward lower failure risk.

> SHAP explains the model's decision. It does not prove that a feature causally produces machine failure.

---

# LLM-Based Maintenance Explanation

The project integrates **Groq** to convert the machine-learning results into a concise explanation for a maintenance engineer.

The LLM receives:

- Failure probability
- Failure prediction
- Anomaly flag
- Anomaly score
- Maintenance status
- Top SHAP contributors

The LLM does **not** make the machine-failure prediction.

The architecture is:

```text
Machine Learning Models
          ↓
      ML Results
          ↓
        SHAP
          ↓
    Groq LLM
          ↓
Human-readable explanation
```

The LLM is instructed to:

- Not override the ML prediction
- Not invent sensor readings
- Distinguish failure prediction from anomaly detection
- Mention important SHAP contributors
- Avoid claiming SHAP proves causation
- Provide practical maintenance guidance
- Keep the explanation concise

---

# Project Architecture

```text
predictive-maintainance-ai/
│
├── .env
├── .gitignore
├── README.md
│
├── train_models.py
├── test_models.py
├── test_pipeline.py
│
├── models/
│   ├── xgb_engineered.joblib
│   └── isolation_forest.joblib
│
└── src/
    ├── config.py
    ├── feature_engineering.py
    ├── prediction.py
    ├── explainability.py
    ├── llm_explanation.py
    └── pipeline.py
```

---

# Project Modules

## `train_models.py`

Trains and saves the final machine-learning models.

It:

1. Loads the dataset
2. Creates engineered features
3. Performs the train-test split
4. Trains XGBoost
5. Trains Isolation Forest
6. Saves the trained models using Joblib

Models are stored in:

```text
models/
```

---

## `src/config.py`

Contains project configuration such as:

- Dataset path
- Model paths
- Base features
- XGBoost parameters
- Isolation Forest contamination
- Failure threshold

---

## `src/feature_engineering.py`

Contains the feature-engineering pipeline used during both training and prediction.

This ensures that training and inference use the same feature representation.

---

## `src/prediction.py`

Provides:

```python
load_models()
```

and:

```python
predict_machine()
```

It handles:

- Model loading
- Feature engineering
- Failure prediction
- Anomaly detection
- Maintenance status generation

---

## `src/explainability.py`

Handles SHAP-based explainability.

Main functions:

```python
create_explainer()
get_shap_values()
get_top_contributors()
```

---

## `src/llm_explanation.py`

Handles Groq-based natural-language maintenance explanations.

It reads the API key from environment variables rather than hardcoding credentials.

---

## `src/pipeline.py`

Provides the main reusable inference function:

```python
analyze_machine(sensor_data)
```

This combines the complete pipeline:

```text
Prediction
    ↓
Anomaly Detection
    ↓
SHAP
    ↓
LLM Explanation
```

Instead of manually executing each step, the application can simply call:

```python
from src.pipeline import analyze_machine

result = analyze_machine(sensor_data)
```

---

# 🧪 Testing the Complete Pipeline

A complete test script is available:

```text
test_pipeline.py
```

Run:

```bash
python test_pipeline.py
```

Example successful result:

```text
==================================================
PREDICTIVE MAINTENANCE RESULT
==================================================
Failure Probability : 0.52%
Failure Prediction  : 0
Anomaly             : No
Anomaly Score       : 0.1740
Maintenance Status  : Normal
```

The same test also successfully generated:

- SHAP top contributors
- Groq maintenance explanation

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Never commit the `.env` file to GitHub.

Add this to `.gitignore`:

```text
.env
```

---

# Tech Stack

### Programming

- Python

### Machine Learning

- XGBoost
- Scikit-learn
- SHAP

### Data Processing

- Pandas
- NumPy

### Model Persistence

- Joblib

### LLM

- Groq API

### Development

- VS Code
- Python Virtual Environment
- Jupyter Notebook

---

# Example Input

```python
sensor_data = {
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 100
}
```

Run the complete pipeline:

```python
from src.pipeline import analyze_machine

result = analyze_machine(sensor_data)

print(result["failure_probability"])
print(result["maintenance_status"])
print(result["top_contributors"])
print(result["explanation"])
```

---

# Example Output

```text
Failure Probability : 0.52%
Failure Prediction  : 0
Anomaly             : No
Anomaly Score       : 0.1740
Maintenance Status  : Normal
```

### Top SHAP Contributors

```text
tool_wear
mechanical_power_proxy
rotational_speed
torque_speed_ratio
temperature_ratio
```

The system then generates a human-readable maintenance explanation using Groq.

---

# End-to-End Workflow

```text
                 ┌───────────────────┐
                 │   Sensor Inputs   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Feature Engineering│
                 └─────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       ┌─────────────┐          ┌───────────────┐
       │   XGBoost   │          │ Isolation     │
       │ Failure     │          │ Forest        │
       │ Prediction  │          │ Anomaly       │
       └──────┬──────┘          └───────┬───────┘
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Maintenance Status│
                 └─────────┬─────────┘
                           │
                           ▼
                    ┌────────────┐
                    │    SHAP    │
                    │Explanation │
                    └─────┬──────┘
                          │
                          ▼
                    ┌────────────┐
                    │   Groq     │
                    │    LLM     │
                    └─────┬──────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ Maintenance Explanation│
              └────────────────────────┘
```

---

# Future Improvements

Potential future extensions include:

- FastAPI REST API
- Interactive web dashboard
- Real-time sensor input
- Prediction history
- Maintenance alerts
- Model performance monitoring
- Database integration
- Cloud deployment
- Real-time machine monitoring
- Improved threshold optimization
- Model retraining pipeline

---

# Author

**Vaishnavi Raskar**

B.E. Computer Engineering  
Pune Institute of Computer Technology (PICT)

---

## Project Summary

This project demonstrates an end-to-end **Explainable AI Predictive Maintenance System** that combines supervised learning, anomaly detection, explainable AI, and LLM-based communication.

The system does not only answer:

> **"Will the machine fail?"**

It also answers:

> **"Is the machine behaving unusually?"**

and:

> **"Which features influenced the prediction?"**

and finally:

> **"How should the result be communicated to a maintenance engineer?"**
