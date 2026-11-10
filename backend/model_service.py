"""Load CKD artifacts once and expose TabNet prediction helpers."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier


PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
PROCESSED_DIR = PROJECT_DIR / "processed"
NUMERICAL_COLUMNS = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
CATEGORICAL_COLUMNS = [
    "sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad",
    "appet", "pe", "ane",
]
DISPLAY_LABELS = {
    "age": "Age (years)", "bp": "Blood Pressure (mm/Hg)", "sg": "Specific Gravity",
    "al": "Albumin", "su": "Sugar", "rbc": "Red Blood Cells", "pc": "Pus Cell",
    "pcc": "Pus Cell Clumps", "ba": "Bacteria", "bgr": "Blood Glucose Random (mgs/dl)",
    "bu": "Blood Urea (mgs/dl)", "sc": "Serum Creatinine (mgs/dl)",
    "sod": "Sodium (mEq/L)", "pot": "Potassium (mEq/L)", "hemo": "Hemoglobin (gms)",
    "pcv": "Packed Cell Volume", "wc": "White Blood Cell Count (cells/cumm)",
    "rc": "Red Blood Cell Count (millions/cmm)", "htn": "Hypertension",
    "dm": "Diabetes Mellitus", "cad": "Coronary Artery Disease", "appet": "Appetite",
    "pe": "Pedal Edema", "ane": "Anemia",
}


def _load_artifacts():
    model = TabNetClassifier(device_name="cpu")
    model.load_model(str(ARTIFACTS_DIR / "tabnet_model.zip"))
    return (
        model,
        joblib.load(ARTIFACTS_DIR / "scaler.pkl"),
        joblib.load(ARTIFACTS_DIR / "encoders.pkl"),
        joblib.load(ARTIFACTS_DIR / "feature_names.pkl"),
    )


MODEL, SCALER, ENCODERS, FEATURE_NAMES = _load_artifacts()


def _encode_and_scale(input_dict: dict) -> np.ndarray:
    """Apply the same categorical encoding and numerical scaling as app.py."""
    row = pd.DataFrame([input_dict])
    for column in CATEGORICAL_COLUMNS:
        classes = ENCODERS[column].classes_
        # sg/al/su were encoded from float values. Their classes_ array has
        # dtype=object, so inspect an actual class rather than its array dtype.
        if isinstance(classes[0], (int, float, np.integer, np.floating)):
            row[column] = pd.to_numeric(row[column], errors="raise")
        row[column] = ENCODERS[column].transform(row[column])
    row[NUMERICAL_COLUMNS] = SCALER.transform(row[NUMERICAL_COLUMNS])
    return row[FEATURE_NAMES].to_numpy(dtype=np.float32)


def _risk_level(prediction: str, ckd_probability: float) -> str:
    if prediction == "notckd" or ckd_probability < 0.60:
        return "Low Risk"
    if ckd_probability < 0.85:
        return "Moderate Risk"
    return "High Risk"


def predict(input_dict: dict) -> dict:
    """Return the decoded prediction, confidence, risk level, and top attributions."""
    X = _encode_and_scale(input_dict)
    probabilities = MODEL.predict_proba(X)[0]
    encoded_prediction = int(np.argmax(probabilities))
    prediction = ENCODERS["classification"].inverse_transform([encoded_prediction])[0]
    ckd_index = int(np.where(ENCODERS["classification"].classes_ == "ckd")[0][0])
    attributions, _ = MODEL.explain(X)
    top_indices = np.argsort(np.abs(attributions[0]))[-3:][::-1]
    return {
        "prediction": prediction,
        "confidence": float(probabilities[encoded_prediction]),
        "risk_level": _risk_level(prediction, float(probabilities[ckd_index])),
        "top_features": [
            {"feature_name": DISPLAY_LABELS[FEATURE_NAMES[index]], "attribution": float(attributions[0][index])}
            for index in top_indices
        ],
    }


def get_global_feature_importance() -> list[dict]:
    """Return descending global attention importances with readable feature names."""
    # TabNet does not persist feature_importances_ in a saved model archive.
    # Recreate a global equivalent from all held-out rows when it is absent.
    if hasattr(MODEL, "feature_importances_"):
        importances = MODEL.feature_importances_
    else:
        reference_data = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
        masks, _ = MODEL.explain(reference_data)
        importances = np.mean(np.abs(masks), axis=0)
    order = np.argsort(importances)[::-1]
    return [
        {"feature_name": DISPLAY_LABELS[FEATURE_NAMES[index]], "importance": float(importances[index])}
        for index in order
    ]
