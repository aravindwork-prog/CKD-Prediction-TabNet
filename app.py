"""Streamlit frontend for the CKD FastAPI prediction service."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://localhost:8000"
NUMERICAL_COLUMNS = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
CATEGORICAL_COLUMNS = ["sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
DISPLAY_LABELS = {
    "age": "Age (years)", "bp": "Blood Pressure (mm/Hg)", "sg": "Specific Gravity", "al": "Albumin", "su": "Sugar", "rbc": "Red Blood Cells", "pc": "Pus Cell", "pcc": "Pus Cell Clumps", "ba": "Bacteria", "bgr": "Blood Glucose Random (mgs/dl)", "bu": "Blood Urea (mgs/dl)", "sc": "Serum Creatinine (mgs/dl)", "sod": "Sodium (mEq/L)", "pot": "Potassium (mEq/L)", "hemo": "Hemoglobin (gms)", "pcv": "Packed Cell Volume", "wc": "White Blood Cell Count (cells/cumm)", "rc": "Red Blood Cell Count (millions/cmm)", "htn": "Hypertension", "dm": "Diabetes Mellitus", "cad": "Coronary Artery Disease", "appet": "Appetite", "pe": "Pedal Edema", "ane": "Anemia",
}
DISPLAY_TO_COLUMN = {label: column for column, label in DISPLAY_LABELS.items()}
CATEGORICAL_OPTIONS = {
    "sg": [1.005, 1.01, 1.015, 1.02, 1.025], "al": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "su": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "rbc": ["abnormal", "normal"], "pc": ["abnormal", "normal"], "pcc": ["not present", "present"], "ba": ["not present", "present"], "htn": ["no", "yes"], "dm": ["no", "yes"], "cad": ["no", "yes"], "appet": ["good", "poor"], "pe": ["no", "yes"], "ane": ["no", "yes"],
}
DEFAULTS = {"age": 51.56, "bp": 76.58, "bgr": 145.06, "bu": 56.69, "sc": 3.0, "sod": 137.63, "pot": 4.58, "hemo": 12.54, "pcv": 39.08, "wc": 8298.5, "rc": 4.74}
TRAINING_MEDIANS = {"age": 48.0, "bp": 80.0, "bgr": 121.0, "bu": 42.0, "sc": 1.3, "sod": 137.0, "pot": 4.4, "hemo": 12.5, "pcv": 39.0, "wc": 8000.0, "rc": 4.7}


def api_request(method: str, path: str, **kwargs):
    """Call the local API and show a clear UI error when it is unavailable."""
    try:
        response = requests.request(method, f"{API_BASE_URL}{path}", timeout=10, **kwargs)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        st.error("Backend not reachable. Make sure the FastAPI server is running on port 8000.")
        return None


def reset_form() -> None:
    """Clear widget state so the form returns to its configured defaults."""
    for column in NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS:
        st.session_state.pop(f"input_{column}", None)
    st.session_state.pop("last_prediction", None)


def build_form() -> tuple[dict, bool, bool]:
    """Render clinical inputs and return their values plus explicit button states."""
    values = {}
    with st.form("ckd_prediction_form"):
        st.subheader("Clinical measurements")
        numeric_left, numeric_right = st.columns(2)
        for index, column in enumerate(NUMERICAL_COLUMNS):
            container = numeric_left if index % 2 == 0 else numeric_right
            values[column] = container.number_input(DISPLAY_LABELS[column], value=float(DEFAULTS[column]), step=0.1, key=f"input_{column}")

        st.subheader("Clinical categories")
        category_left, category_right = st.columns(2)
        for index, column in enumerate(CATEGORICAL_COLUMNS):
            container = category_left if index % 2 == 0 else category_right
            values[column] = container.selectbox(DISPLAY_LABELS[column], CATEGORICAL_OPTIONS[column], key=f"input_{column}")

        predict_column, reset_column = st.columns(2)
        submitted = predict_column.form_submit_button("Predict")
        reset = reset_column.form_submit_button("Reset Form", on_click=reset_form)
    return values, submitted, reset


def explanation_sentence(top_features: list[dict], values: dict) -> str:
    """Describe the top two API attributions in the existing plain-language style."""
    phrases = []
    for feature in top_features[:2]:
        label = feature["feature_name"]
        column = DISPLAY_TO_COLUMN[label]
        if column in NUMERICAL_COLUMNS:
            direction = "elevated" if float(values[column]) >= TRAINING_MEDIANS[column] else "low"
            phrases.append(f"{direction} {label}")
        else:
            phrases.append(label)
    return f"This prediction was primarily driven by {phrases[0]} and {phrases[1]}."


def display_prediction(result: dict, values: dict) -> None:
    """Display API prediction data with the existing risk and explanation visuals."""
    outcome = "CKD Positive" if result["prediction"] == "ckd" else "CKD Negative"
    risk_colors = {"High Risk": "#c62828", "Moderate Risk": "#c78b00", "Low Risk": "#198754"}
    result_column, risk_column = st.columns(2)
    result_column.success(outcome)
    result_column.metric("Confidence", f"{result['confidence']:.2%}")
    risk = result["risk_level"]
    risk_column.markdown(f"<div style='margin-top:0.5rem; font-weight:700; color:white; background:{risk_colors[risk]}; padding:0.5rem 0.75rem; border-radius:0.4rem; text-align:center'>{risk}</div>", unsafe_allow_html=True)

    top_features = result["top_features"]
    names = np.array([item["feature_name"] for item in top_features])
    scores = np.array([item["attribution"] for item in top_features])
    order = np.argsort(scores)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names[order], scores[order], color="teal")
    ax.set_xlabel("Attention-based attribution")
    ax.set_title("Features driving this prediction")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(explanation_sentence(top_features, values))


def render_sidebar() -> None:
    """Render live global importance and recent backend prediction history."""
    with st.sidebar:
        st.header("Model Insights")
        importance = api_request("GET", "/feature-importance")
        if importance:
            frame = pd.DataFrame(importance)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(frame["feature_name"][::-1], frame["importance"][::-1], color="teal")
            ax.set_xlabel("Global feature importance")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.header("Recent Predictions")
        history = api_request("GET", "/history?limit=5")
        if history:
            table = pd.DataFrame(history)[["timestamp", "prediction", "confidence", "risk_level"]]
            st.dataframe(table, use_container_width=True, hide_index=True)
        elif history == []:
            st.caption("No predictions logged yet.")


def main() -> None:
    """Run the Streamlit CKD frontend."""
    st.set_page_config(page_title="CKD Predictor", page_icon="🩺", layout="wide")
    st.title("Chronic Kidney Disease Predictor")
    st.caption("Enter clinical values to receive a TabNet-based CKD prediction.")
    st.info("For preliminary screening and educational purposes only. This tool is not a medical diagnosis; consult a qualified healthcare professional for diagnosis and treatment.")
    values, submitted, reset = build_form()

    # A POST is permitted only on the exact rerun where Predict was clicked.
    if submitted and not reset:
        result = api_request("POST", "/predict", json=values)
        if result:
            st.session_state["last_prediction"] = {"result": result, "values": values}

    # Preserve the last deliberate result during unrelated Streamlit reruns.
    if "last_prediction" in st.session_state:
        saved_prediction = st.session_state["last_prediction"]
        display_prediction(saved_prediction["result"], saved_prediction["values"])

    # Render after a POST so this table includes that newly logged prediction.
    render_sidebar()


if __name__ == "__main__":
    main()
