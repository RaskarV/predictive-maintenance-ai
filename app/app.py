# import sys
# from pathlib import Path

# import pandas as pd
# import streamlit as st


# # ============================================================
# # PROJECT PATH SETUP
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.append(str(PROJECT_ROOT))


# # ============================================================
# # IMPORT ML PIPELINE
# # ============================================================

# from src.pipeline import analyze_machine


# # ============================================================
# # DATASET PATH
# # ============================================================

# DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


# # ============================================================
# # STREAMLIT CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title="AI Predictive Maintenance",
#     page_icon="⚙️",
#     layout="wide"
# )


# # ============================================================
# # LOAD HISTORICAL DATA
# # ============================================================

# @st.cache_data
# def load_historical_data():

#     return pd.read_csv(DATA_PATH)


# df = load_historical_data()


# # ============================================================
# # HEADER
# # ============================================================

# st.title("⚙️ AI Predictive Maintenance System")

# st.markdown(
#     """
#     **Predict machine failure • Detect unusual behavior • Explain predictions**

#     An end-to-end predictive maintenance system using
#     XGBoost, Isolation Forest, SHAP, and Groq.
#     """
# )


# # ============================================================
# # SIDEBAR
# # ============================================================

# st.sidebar.title("⚙️ Analysis Mode")

# analysis_mode = st.sidebar.radio(
#     "Choose analysis mode:",
#     [
#         "🧑‍💻 Manual Sensor Input",
#         "🏭 Historical Machine"
#     ]
# )


# # ============================================================
# # COMMON FUNCTION — DISPLAY RESULTS
# # ============================================================

# def display_results(result):

#     # --------------------------------------------------------
#     # Extract results
#     # --------------------------------------------------------

#     failure_probability = result["failure_probability"]
#     failure_prediction = result["failure_prediction"]

#     anomaly_flag = result["anomaly_flag"]
#     anomaly_score = result["anomaly_score"]

#     maintenance_status = result["maintenance_status"]

#     top_contributors = result["top_contributors"]

#     explanation = result["explanation"]


#     # ========================================================
#     # MACHINE ASSESSMENT
#     # ========================================================

#     st.divider()

#     st.subheader("📊 Machine Assessment")

#     col1, col2, col3 = st.columns(3)


#     with col1:

#         st.metric(
#             "Failure Probability",
#             f"{failure_probability * 100:.2f}%"
#         )


#     with col2:

#         if anomaly_flag == 1:
#             anomaly_text = "⚠️ Anomaly"
#         else:
#             anomaly_text = "✅ Normal"

#         st.metric(
#             "Anomaly Detection",
#             anomaly_text
#         )


#     with col3:

#         st.metric(
#             "Maintenance Status",
#             maintenance_status
#         )


#     # ========================================================
#     # FAILURE PREDICTION
#     # ========================================================

#     st.divider()

#     if failure_prediction == 1:

#         st.error(
#             f"⚠️ Machine failure risk detected "
#             f"({failure_probability * 100:.2f}% probability)."
#         )

#     else:

#         st.success(
#             f"✅ No failure predicted "
#             f"({failure_probability * 100:.2f}% probability)."
#         )


#     # ========================================================
#     # ANOMALY
#     # ========================================================

#     if anomaly_flag == 1:

#         st.warning(
#             f"⚠️ Isolation Forest detected unusual sensor behavior. "
#             f"Anomaly score: {anomaly_score:.4f}"
#         )

#     else:

#         st.info(
#             f"Normal sensor behavior detected. "
#             f"Anomaly score: {anomaly_score:.4f}"
#         )


#     # ========================================================
#     # SHAP EXPLANATION
#     # ========================================================

#     st.divider()

#     st.subheader("🔎 Prediction Explanation (SHAP)")

#     st.markdown(
#         """
#         SHAP explains how each feature influenced the model's
#         prediction for this particular machine.
#         """
#     )


#     # --------------------------------------------------------
#     # Prepare chart data
#     # --------------------------------------------------------

#     chart_data = top_contributors.copy()

#     chart_data = chart_data.sort_values(
#         by="shap_value",
#         ascending=True
#     )


#     # --------------------------------------------------------
#     # SHAP chart
#     # --------------------------------------------------------

#     st.markdown("#### 📊 Feature Contributions")

#     st.bar_chart(
#         chart_data.set_index("feature")["shap_value"],
#         horizontal=True
#     )


#     st.caption(
#         "Negative SHAP values push the model output toward a "
#         "lower failure prediction. Positive SHAP values push "
#         "the model output toward a higher failure prediction."
#     )


#     # ========================================================
#     # SHAP TABLE
#     # ========================================================

#     st.markdown("#### 🔬 Contributor Details")

#     display_contributors = top_contributors.copy()

#     display_contributors["impact"] = (
#         display_contributors["shap_value"]
#         .apply(
#             lambda value:
#             "Positive contribution to model output"
#             if value > 0
#             else "Negative contribution to model output"
#         )
#     )

#     display_contributors = display_contributors.rename(
#         columns={
#             "feature": "Feature",
#             "feature_value": "Feature Value",
#             "shap_value": "SHAP Value",
#             "impact": "Impact"
#         }
#     )

#     st.dataframe(
#         display_contributors,
#         use_container_width=True,
#         hide_index=True
#     )


#     # ========================================================
#     # GROQ EXPLANATION
#     # ========================================================

#     st.divider()

#     st.subheader("🤖 AI Maintenance Explanation")

#     if explanation:

#         st.markdown(explanation)

#     else:

#         st.warning(
#             "AI maintenance explanation could not be generated."
#         )


#     # ========================================================
#     # TECHNICAL DETAILS
#     # ========================================================

#     st.divider()

#     with st.expander("🧠 View Technical Details"):

#         st.write(
#             "**Failure Prediction:**",
#             failure_prediction
#         )

#         st.write(
#             "**Failure Probability:**",
#             f"{failure_probability:.6f}"
#         )

#         st.write(
#             "**Anomaly Flag:**",
#             anomaly_flag
#         )

#         st.write(
#             "**Anomaly Score:**",
#             f"{anomaly_score:.6f}"
#         )

#         st.write(
#             "**Maintenance Status:**",
#             maintenance_status
#         )

#         st.markdown(
#             """
#             **Models used**

#             - XGBoost → failure prediction
#             - Isolation Forest → anomaly detection
#             - SHAP → prediction explainability
#             - Groq → human-readable maintenance explanation
#             """
#         )


# # ============================================================
# # MODE 1 — MANUAL SENSOR INPUT
# # ============================================================

# if analysis_mode == "🧑‍💻 Manual Sensor Input":

#     st.subheader("🔧 Manual Sensor Inputs")

#     col1, col2 = st.columns(2)


#     # --------------------------------------------------------
#     # LEFT COLUMN
#     # --------------------------------------------------------

#     with col1:

#         air_temperature = st.number_input(
#             "Air Temperature [K]",
#             min_value=290.0,
#             max_value=310.0,
#             value=300.0,
#             step=0.1
#         )

#         rotational_speed = st.number_input(
#             "Rotational Speed [rpm]",
#             min_value=500,
#             max_value=3000,
#             value=1500,
#             step=10
#         )

#         tool_wear = st.number_input(
#             "Tool Wear [min]",
#             min_value=0,
#             max_value=300,
#             value=100,
#             step=1
#         )


#     # --------------------------------------------------------
#     # RIGHT COLUMN
#     # --------------------------------------------------------

#     with col2:

#         process_temperature = st.number_input(
#             "Process Temperature [K]",
#             min_value=295.0,
#             max_value=320.0,
#             value=310.0,
#             step=0.1
#         )

#         torque = st.number_input(
#             "Torque [Nm]",
#             min_value=0.0,
#             max_value=100.0,
#             value=40.0,
#             step=0.5
#         )


#     # --------------------------------------------------------
#     # ANALYZE BUTTON
#     # --------------------------------------------------------

#     st.divider()

#     if st.button(
#         "🔍 Analyze Machine",
#         type="primary",
#         use_container_width=True
#     ):

#         sensor_data = {

#             "Air temperature [K]":
#                 air_temperature,

#             "Process temperature [K]":
#                 process_temperature,

#             "Rotational speed [rpm]":
#                 rotational_speed,

#             "Torque [Nm]":
#                 torque,

#             "Tool wear [min]":
#                 tool_wear
#         }


#         with st.spinner(
#             "Analyzing machine condition..."
#         ):

#             try:

#                 result = analyze_machine(
#                     sensor_data
#                 )

#                 display_results(result)

#             except Exception as error:

#                 st.error(
#                     "An error occurred while analyzing the machine."
#                 )

#                 st.exception(error)


# # ============================================================
# # MODE 2 — HISTORICAL MACHINE
# # ============================================================

# else:

#     st.subheader("🏭 Historical Machine Analysis")

#     st.markdown(
#         """
#         Select a machine from the **AI4I 2020 Predictive Maintenance
#         Dataset** and run the same production inference pipeline
#         used for manual sensor input.
#         """
#     )


#     # --------------------------------------------------------
#     # MACHINE SELECTION
#     # --------------------------------------------------------

#     machine_ids = df["UDI"].tolist()

#     selected_udi = st.selectbox(
#         "Select Machine (UDI)",
#         machine_ids
#     )


#     # --------------------------------------------------------
#     # GET SELECTED MACHINE
#     # --------------------------------------------------------

#     selected_machine = df[
#         df["UDI"] == selected_udi
#     ].iloc[0]


#     # ========================================================
#     # HISTORICAL MACHINE INFORMATION
#     # ========================================================

#     st.markdown("#### 📋 Machine Information")

#     info_col1, info_col2, info_col3 = st.columns(3)


#     with info_col1:

#         st.metric(
#             "UDI",
#             int(selected_machine["UDI"])
#         )


#     with info_col2:

#         st.metric(
#             "Product ID",
#             selected_machine["Product ID"]
#         )


#     with info_col3:

#         machine_type = selected_machine["Type"]

#         st.metric(
#             "Machine Type",
#             machine_type
#         )


#     # ========================================================
#     # SENSOR VALUES
#     # ========================================================

#     st.markdown("#### 📡 Sensor Readings")

#     sensor_col1, sensor_col2 = st.columns(2)


#     with sensor_col1:

#         st.metric(
#             "Air Temperature",
#             f"{selected_machine['Air temperature [K]']:.2f} K"
#         )

#         st.metric(
#             "Rotational Speed",
#             f"{selected_machine['Rotational speed [rpm]']:.0f} rpm"
#         )

#         st.metric(
#             "Tool Wear",
#             f"{selected_machine['Tool wear [min]']:.0f} min"
#         )


#     with sensor_col2:

#         st.metric(
#             "Process Temperature",
#             f"{selected_machine['Process temperature [K]']:.2f} K"
#         )

#         st.metric(
#             "Torque",
#             f"{selected_machine['Torque [Nm]']:.2f} Nm"
#         )


#     # ========================================================
#     # ACTUAL HISTORICAL OUTCOME
#     # ========================================================

#     actual_failure = int(
#         selected_machine["Machine failure"]
#     )


#     if actual_failure == 1:

#         st.warning(
#             "Historical dataset outcome: Machine failure = 1"
#         )

#     else:

#         st.success(
#             "Historical dataset outcome: Machine failure = 0"
#         )


#     # ========================================================
#     # ANALYZE HISTORICAL MACHINE
#     # ========================================================

#     st.divider()

#     if st.button(
#         "🔍 Analyze Historical Machine",
#         type="primary",
#         use_container_width=True
#     ):


#         # ----------------------------------------------------
#         # IMPORTANT:
#         # UDI, Product ID, Type and failure indicators
#         # are NOT passed into the ML model.
#         #
#         # Only the five predictive sensor features are used.
#         # ----------------------------------------------------

#         sensor_data = {

#             "Air temperature [K]":
#                 selected_machine["Air temperature [K]"],

#             "Process temperature [K]":
#                 selected_machine["Process temperature [K]"],

#             "Rotational speed [rpm]":
#                 selected_machine["Rotational speed [rpm]"],

#             "Torque [Nm]":
#                 selected_machine["Torque [Nm]"],

#             "Tool wear [min]":
#                 selected_machine["Tool wear [min]"]
#         }


#         with st.spinner(
#             "Running historical machine through the ML pipeline..."
#         ):

#             try:

#                 result = analyze_machine(
#                     sensor_data
#                 )

#                 display_results(result)

#             except Exception as error:

#                 st.error(
#                     "An error occurred while analyzing the machine."
#                 )

#                 st.exception(error)


# # ============================================================
# # FOOTER
# # ============================================================

# st.divider()

# st.caption(
#     "AI-Powered Predictive Maintenance & Anomaly Detection System"
# )
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORT ML PIPELINE
# ============================================================

from src.pipeline import analyze_machine


# ============================================================
# DATASET PATH
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# GLOBAL CUSTOM CSS
# ============================================================

st.html("""
<style>

html, body {
    font-family: "Segoe UI", sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #f5f8fc 0%,
        #eef4f9 50%,
        #f8fbfd 100%
    );
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ---------- HERO ---------- */

.hero {
    background: linear-gradient(
        135deg,
        #102a43 0%,
        #174a6e 52%,
        #167d8d 100%
    );

    padding: 35px 40px;
    border-radius: 22px;
    margin-bottom: 30px;

    box-shadow:
        0 12px 35px rgba(16, 42, 67, 0.20);

    color: white;
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 8px;
    color: white;
}

.hero-subtitle {
    font-size: 17px;
    color: #d9edf2;
    line-height: 1.5;
    margin-bottom: 20px;
}

.hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.hero-badge {
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    padding: 8px 14px;
    border-radius: 20px;
    font-size: 13px;
    color: #f5ffff;
}


/* ---------- SECTION ---------- */

.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #102a43;
    margin-top: 10px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #627d98;
    font-size: 14px;
    margin-bottom: 18px;
}


/* ---------- MODE CARDS ---------- */

.mode-card {
    background: white;
    border: 1px solid #d9e2ec;
    border-radius: 18px;
    padding: 22px;
    min-height: 120px;
    box-shadow: 0 5px 18px rgba(16,42,67,0.07);
}

.mode-icon {
    font-size: 30px;
    margin-bottom: 8px;
}

.mode-title {
    color: #102a43;
    font-size: 18px;
    font-weight: 800;
}

.mode-description {
    color: #627d98;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 6px;
}


/* ---------- INPUT CARD ---------- */

.input-card {
    background: white;
    border: 1px solid #d9e2ec;
    border-radius: 18px;
    padding: 22px;
    margin-top: 12px;
    margin-bottom: 18px;
    box-shadow: 0 5px 18px rgba(16,42,67,0.06);
}

.input-title {
    color: #102a43;
    font-size: 18px;
    font-weight: 800;
}

.input-description {
    color: #627d98;
    font-size: 13px;
    margin-top: 5px;
}


/* ---------- RESULT CARDS ---------- */

.result-card {
    background: white;
    border: 1px solid #d9e2ec;
    border-radius: 18px;
    padding: 20px;
    min-height: 135px;
    box-shadow: 0 7px 22px rgba(16,42,67,0.08);
}

.result-label {
    color: #627d98;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.result-value {
    color: #102a43;
    font-size: 25px;
    font-weight: 800;
    margin-top: 10px;
    line-height: 1.25;
}

.result-description {
    color: #829ab1;
    font-size: 12px;
    margin-top: 8px;
}

.status-normal {
    border-left: 6px solid #2eaf7d;
}

.status-unusual {
    border-left: 6px solid #e5a93d;
}

.status-failure {
    border-left: 6px solid #e67e22;
}

.status-high {
    border-left: 6px solid #d64545;
}


/* ---------- ALERTS ---------- */

.custom-alert {
    border-radius: 15px;
    padding: 17px 20px;
    margin: 14px 0;
    font-size: 14px;
    line-height: 1.6;
}

.alert-success {
    background: #edf9f4;
    border: 1px solid #b9ead5;
    color: #176b4d;
}

.alert-warning {
    background: #fff8e8;
    border: 1px solid #f3d99b;
    color: #855f16;
}

.alert-danger {
    background: #fff0ef;
    border: 1px solid #efc1bd;
    color: #8c2f2b;
}

.alert-info {
    background: #eef7fb;
    border: 1px solid #c7e5ef;
    color: #185c70;
}


/* ---------- SHAP ---------- */

.shap-header {
    background: white;
    border: 1px solid #d9e2ec;
    border-radius: 18px;
    padding: 22px;
    margin-top: 24px;
    margin-bottom: 15px;
    box-shadow: 0 5px 18px rgba(16,42,67,0.06);
}

.shap-title {
    color: #102a43;
    font-size: 21px;
    font-weight: 800;
}

.shap-description {
    color: #627d98;
    font-size: 13px;
    margin-top: 5px;
}


/* ---------- AI ---------- */

.ai-card {
    background: linear-gradient(
        135deg,
        #eef9fb,
        #f7fcfd
    );

    border: 1px solid #b9e1e8;
    border-radius: 18px;
    padding: 23px;
    margin-top: 24px;
    margin-bottom: 15px;

    box-shadow:
        0 7px 22px rgba(22,125,141,0.08);
}

.ai-title {
    color: #105b6d;
    font-size: 21px;
    font-weight: 800;
}

.ai-description {
    color: #557c88;
    font-size: 13px;
    margin-top: 5px;
}


/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: #829ab1;
    font-size: 12px;
    line-height: 1.7;
    padding-top: 25px;
}


/* ---------- BUTTON ---------- */

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    min-height: 45px;
}

</style>
""")


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

@st.cache_data
def load_historical_data():
    return pd.read_csv(DATA_PATH)


df = load_historical_data()


# ============================================================
# HERO HEADER
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-title">
        ⚙️ AI Predictive Maintenance
    </div>

    <div class="hero-subtitle">
        Intelligent machine health monitoring powered by
        machine learning, anomaly detection and explainable AI.
    </div>

    <div class="hero-badges">
        <div class="hero-badge">📈 Failure Prediction</div>
        <div class="hero-badge">🔍 Anomaly Detection</div>
        <div class="hero-badge">🧠 SHAP Explainability</div>
        <div class="hero-badge">🤖 AI Maintenance Assistant</div>
    </div>

</div>
""")


# ============================================================
# ANALYSIS MODE
# ============================================================

st.markdown("### Choose Analysis Mode")

st.markdown(
    "Analyze a machine using current sensor inputs or historical industrial data."
)


if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "manual"


mode_col1, mode_col2 = st.columns(2)


with mode_col1:

    st.html("""
    <div class="mode-card">

        <div class="mode-icon">🧑‍💻</div>

        <div class="mode-title">
            Manual Sensor Analysis
        </div>

        <div class="mode-description">
            Enter current machine sensor readings
            and evaluate its health condition.
        </div>

    </div>
    """)

    if st.button(
        "Use Manual Analysis",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.analysis_mode == "manual"
            else "secondary"
        )
    ):
        st.session_state.analysis_mode = "manual"
        st.rerun()


with mode_col2:

    st.html("""
    <div class="mode-card">

        <div class="mode-icon">🏭</div>

        <div class="mode-title">
            Historical Machine Analysis
        </div>

        <div class="mode-description">
            Select a machine from the AI4I dataset
            and reproduce its historical sensor condition.
        </div>

    </div>
    """)

    if st.button(
        "Use Historical Analysis",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.analysis_mode == "historical"
            else "secondary"
        )
    ):
        st.session_state.analysis_mode = "historical"
        st.rerun()


analysis_mode = st.session_state.analysis_mode


# ============================================================
# COMMON RESULT FUNCTION
# ============================================================

def display_results(result):

    failure_probability = result["failure_probability"]
    failure_prediction = result["failure_prediction"]

    anomaly_flag = result["anomaly_flag"]
    anomaly_score = result["anomaly_score"]

    maintenance_status = result["maintenance_status"]

    top_contributors = result["top_contributors"]

    explanation = result["explanation"]


    # ========================================================
    # STATUS STYLE
    # ========================================================

    if maintenance_status == "Normal":
        status_class = "status-normal"
        status_icon = "🟢"

    elif maintenance_status == "Unusual Behavior":
        status_class = "status-unusual"
        status_icon = "🟡"

    elif maintenance_status == "Predicted Failure":
        status_class = "status-failure"
        status_icon = "🟠"

    else:
        status_class = "status-high"
        status_icon = "🔴"


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.markdown("### 📊 Machine Health Assessment")

    st.markdown(
        "AI-generated assessment from the predictive maintenance pipeline."
    )


    # ========================================================
    # RESULT CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.html(f"""
        <div class="result-card">

            <div class="result-label">
                Failure Probability
            </div>

            <div class="result-value">
                {failure_probability * 100:.2f}%
            </div>

            <div class="result-description">
                XGBoost prediction
            </div>

        </div>
        """)


    with col2:

        anomaly_display = (
            "⚠️ Detected"
            if anomaly_flag == 1
            else "✅ Normal"
        )

        st.html(f"""
        <div class="result-card">

            <div class="result-label">
                Anomaly Detection
            </div>

            <div class="result-value">
                {anomaly_display}
            </div>

            <div class="result-description">
                Isolation Forest
            </div>

        </div>
        """)


    with col3:

        prediction_display = (
            "⚠️ Failure"
            if failure_prediction == 1
            else "✅ No Failure"
        )

        st.html(f"""
        <div class="result-card">

            <div class="result-label">
                Prediction
            </div>

            <div class="result-value">
                {prediction_display}
            </div>

            <div class="result-description">
                XGBoost decision
            </div>

        </div>
        """)


    with col4:

        st.html(f"""
        <div class="result-card {status_class}">

            <div class="result-label">
                Maintenance Status
            </div>

            <div class="result-value">
                {status_icon} {maintenance_status}
            </div>

            <div class="result-description">
                Combined assessment
            </div>

        </div>
        """)


    # ========================================================
    # FAILURE MESSAGE
    # ========================================================

    if failure_prediction == 1:

        st.html(f"""
        <div class="custom-alert alert-danger">

            <strong>⚠️ Failure Risk Detected</strong><br>

            The model predicts a machine failure probability of
            <strong>{failure_probability * 100:.2f}%</strong>.

        </div>
        """)

    else:

        st.html(f"""
        <div class="custom-alert alert-success">

            <strong>✅ No Failure Predicted</strong><br>

            The model predicts a machine failure probability of
            <strong>{failure_probability * 100:.2f}%</strong>.

        </div>
        """)


    # ========================================================
    # ANOMALY MESSAGE
    # ========================================================

    if anomaly_flag == 1:

        st.html(f"""
        <div class="custom-alert alert-warning">

            <strong>⚠️ Unusual Sensor Behavior Detected</strong><br>

            Isolation Forest identified an unusual observation.
            Anomaly score: <strong>{anomaly_score:.4f}</strong>

        </div>
        """)

    else:

        st.html(f"""
        <div class="custom-alert alert-info">

            <strong>🔵 No Anomaly Detected</strong><br>

            Isolation Forest did not identify unusual sensor behavior.
            Anomaly score: <strong>{anomaly_score:.4f}</strong>

        </div>
        """)


    # ========================================================
    # SHAP HEADER
    # ========================================================

    st.html("""
    <div class="shap-header">

        <div class="shap-title">
            🧠 Why did the model make this prediction?
        </div>

        <div class="shap-description">
            SHAP shows how individual features contributed
            to this machine's model output.
        </div>

    </div>
    """)


    # ========================================================
    # SHAP CHART
    # ========================================================

    st.markdown("#### 📊 Feature Contributions")


    chart_data = top_contributors.copy()

    chart_data = chart_data.sort_values(
        by="shap_value",
        ascending=True
    )


    st.bar_chart(
        chart_data.set_index("feature")["shap_value"],
        horizontal=True
    )


    st.caption(
        "Positive SHAP values push the model output toward a higher "
        "failure prediction. Negative SHAP values push the model "
        "output toward a lower failure prediction."
    )


    # ========================================================
    # SHAP TABLE
    # ========================================================

    st.markdown("#### 🔬 Contributor Details")


    display_contributors = top_contributors.copy()


    display_contributors["impact"] = (
        display_contributors["shap_value"]
        .apply(
            lambda value:
            "Positive contribution to model output"
            if value > 0
            else "Negative contribution to model output"
        )
    )


    display_contributors = display_contributors.rename(
        columns={
            "feature": "Feature",
            "feature_value": "Feature Value",
            "shap_value": "SHAP Value",
            "impact": "Impact"
        }
    )


    st.dataframe(
        display_contributors,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # GROQ AI EXPLANATION
    # ========================================================

    st.html("""
    <div class="ai-card">

        <div class="ai-title">
            🤖 AI Maintenance Assistant
        </div>

        <div class="ai-description">
            Human-readable interpretation generated from the
            ML prediction, anomaly result and SHAP contributors.
        </div>

    </div>
    """)


    if explanation:

        st.markdown(explanation)

    else:

        st.warning(
            "AI maintenance explanation could not be generated."
        )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander("🔧 View Technical Details"):

        tech_col1, tech_col2 = st.columns(2)


        with tech_col1:

            st.write(
                "**Failure Prediction:**",
                failure_prediction
            )

            st.write(
                "**Failure Probability:**",
                f"{failure_probability:.6f}"
            )

            st.write(
                "**Anomaly Flag:**",
                anomaly_flag
            )


        with tech_col2:

            st.write(
                "**Anomaly Score:**",
                f"{anomaly_score:.6f}"
            )

            st.write(
                "**Maintenance Status:**",
                maintenance_status
            )


        st.markdown("### Models Used")

        st.markdown(
            """
            - **XGBoost** → Failure prediction
            - **Isolation Forest** → Anomaly detection
            - **SHAP** → Prediction explainability
            - **Groq** → Human-readable maintenance explanation
            """
        )


# ============================================================
# MODE 1 — MANUAL SENSOR INPUT
# ============================================================

if analysis_mode == "manual":

    st.markdown("### 🧑‍💻 Manual Sensor Analysis")

    st.markdown(
        "Enter the current sensor readings of the machine you want to analyze."
    )


    st.html("""
    <div class="input-card">

        <div class="input-title">
            📡 Machine Sensor Parameters
        </div>

        <div class="input-description">
            Provide the five sensor measurements used by the
            predictive maintenance model.
        </div>

    </div>
    """)


    col1, col2 = st.columns(2)


    with col1:

        air_temperature = st.number_input(
            "🌡️ Air Temperature [K]",
            min_value=290.0,
            max_value=310.0,
            value=300.0,
            step=0.1
        )


        rotational_speed = st.number_input(
            "⚙️ Rotational Speed [rpm]",
            min_value=500,
            max_value=3000,
            value=1500,
            step=10
        )


        tool_wear = st.number_input(
            "🛠️ Tool Wear [min]",
            min_value=0,
            max_value=300,
            value=100,
            step=1
        )


    with col2:

        process_temperature = st.number_input(
            "🌡️ Process Temperature [K]",
            min_value=295.0,
            max_value=320.0,
            value=310.0,
            step=0.1
        )


        torque = st.number_input(
            "🔩 Torque [Nm]",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.5
        )


    st.markdown("")


    if st.button(
        "🔍 Analyze Machine Condition",
        type="primary",
        use_container_width=True
    ):

        sensor_data = {
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }


        with st.spinner(
            "🔄 Running predictive maintenance pipeline..."
        ):

            try:

                result = analyze_machine(sensor_data)

                display_results(result)

            except Exception as error:

                st.error(
                    "An error occurred while analyzing the machine."
                )

                st.exception(error)


# ============================================================
# MODE 2 — HISTORICAL MACHINE
# ============================================================

else:

    st.markdown("### 🏭 Historical Machine Analysis")

    st.markdown(
        "Select a machine from the AI4I 2020 dataset and run its "
        "historical sensor readings through the same inference pipeline."
    )


    # ========================================================
    # MACHINE SELECTION
    # ========================================================

    st.html("""
    <div class="input-card">

        <div class="input-title">
            🗂️ Machine Selection
        </div>

        <div class="input-description">
            Choose a machine using its unique identifier (UDI).
        </div>

    </div>
    """)


    machine_ids = df["UDI"].tolist()


    selected_udi = st.selectbox(
        "Select Machine (UDI)",
        machine_ids
    )


    selected_machine = df[
        df["UDI"] == selected_udi
    ].iloc[0]


    # ========================================================
    # MACHINE INFORMATION
    # ========================================================

    st.markdown("### 📋 Machine Information")


    info_col1, info_col2, info_col3 = st.columns(3)


    with info_col1:

        st.metric(
            "UDI",
            int(selected_machine["UDI"])
        )


    with info_col2:

        st.metric(
            "Product ID",
            selected_machine["Product ID"]
        )


    with info_col3:

        st.metric(
            "Machine Type",
            selected_machine["Type"]
        )


    # ========================================================
    # SENSOR SNAPSHOT
    # ========================================================

    st.markdown("### 📡 Sensor Snapshot")


    sensor_col1, sensor_col2, sensor_col3, sensor_col4, sensor_col5 = (
        st.columns(5)
    )


    with sensor_col1:

        st.metric(
            "🌡 Air Temp",
            f"{selected_machine['Air temperature [K]']:.2f} K"
        )


    with sensor_col2:

        st.metric(
            "🌡 Process Temp",
            f"{selected_machine['Process temperature [K]']:.2f} K"
        )


    with sensor_col3:

        st.metric(
            "⚙ RPM",
            f"{selected_machine['Rotational speed [rpm]']:.0f}"
        )


    with sensor_col4:

        st.metric(
            "🔩 Torque",
            f"{selected_machine['Torque [Nm]']:.2f} Nm"
        )


    with sensor_col5:

        st.metric(
            "🛠 Tool Wear",
            f"{selected_machine['Tool wear [min]']:.0f} min"
        )


    # ========================================================
    # HISTORICAL OUTCOME
    # ========================================================

    actual_failure = int(
        selected_machine["Machine failure"]
    )


    st.markdown("### 📚 Historical Dataset Outcome")


    if actual_failure == 1:

        st.html("""
        <div class="custom-alert alert-warning">

            <strong>⚠️ Historical Label: Machine Failure = 1</strong><br>

            This machine was labelled as a failure case
            in the original dataset.

        </div>
        """)

    else:

        st.html("""
        <div class="custom-alert alert-info">

            <strong>🔵 Historical Label: Machine Failure = 0</strong><br>

            This machine was labelled as a non-failure case
            in the original dataset.

        </div>
        """)


    # ========================================================
    # HISTORICAL ANALYSIS
    # ========================================================

    if st.button(
        "🔍 Analyze Historical Machine",
        type="primary",
        use_container_width=True
    ):

        # IMPORTANT:
        #
        # UDI, Product ID, Type and failure indicators
        # are NOT passed into the ML model.
        #
        # Only the five predictive sensor features are used.

        sensor_data = {
            "Air temperature [K]":
                selected_machine["Air temperature [K]"],

            "Process temperature [K]":
                selected_machine["Process temperature [K]"],

            "Rotational speed [rpm]":
                selected_machine["Rotational speed [rpm]"],

            "Torque [Nm]":
                selected_machine["Torque [Nm]"],

            "Tool wear [min]":
                selected_machine["Tool wear [min]"]
        }


        with st.spinner(
            "🔄 Running historical machine through the ML pipeline..."
        ):

            try:

                result = analyze_machine(sensor_data)

                display_results(result)

            except Exception as error:

                st.error(
                    "An error occurred while analyzing the machine."
                )

                st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    ⚙️ AI-Powered Predictive Maintenance & Anomaly Detection System

    <br>

    XGBoost • Isolation Forest • SHAP • Groq

</div>
""")