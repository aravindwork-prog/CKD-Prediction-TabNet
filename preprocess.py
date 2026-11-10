"""Preprocess the UCI Chronic Kidney Disease dataset for model training."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATA_PATH = Path("data/ckd.csv")
PROCESSED_DIR = Path("processed")
ARTIFACTS_DIR = Path("artifacts")
TARGET_COLUMN = "classification"

# sg, al, and su are deliberately categorical: they are ordinal test readings.
NUMERICAL_COLUMNS = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
CATEGORICAL_COLUMNS = [
    "sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad",
    "appet", "pe", "ane",
]


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV and remove the non-clinical row index."""
    data = pd.read_csv(path)
    data = data.drop(columns="id")
    print("1. Loaded data and dropped the 'id' column.")
    print(f"   Shape: {data.shape}")
    print("   Dtypes:\n", data.dtypes)
    print("   Missing values per column:\n", data.isna().sum())
    return data


def clean_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Trim string values and make equivalent category spellings consistent."""
    data = data.copy()
    object_columns = data.select_dtypes(include=["object", "string"]).columns

    for column in object_columns:
        cleaned = data[column].astype("string").str.strip().str.lower()
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
        # The original data uses both spellings; retain one readable spelling.
        cleaned = cleaned.replace({"notpresent": "not present"})
        # Use NumPy NaN rather than pandas' nullable-string NA for sklearn.
        data[column] = cleaned.astype(object).where(cleaned.notna(), np.nan)

    # Some numeric columns may be read as strings in alternate CKD CSV exports.
    for column in NUMERICAL_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    print("2. Stripped whitespace and standardized category labels.")
    return data


def impute_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Fill numerical values with medians and categorical values with modes."""
    data = data.copy()
    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    data[NUMERICAL_COLUMNS] = numeric_imputer.fit_transform(data[NUMERICAL_COLUMNS])
    data[CATEGORICAL_COLUMNS] = categorical_imputer.fit_transform(data[CATEGORICAL_COLUMNS])

    print("3. Imputed numerical values with medians and categorical values with modes.")
    return data


def encode_categories(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode each categorical feature and the classification target."""
    data = data.copy()
    encoders: dict[str, LabelEncoder] = {}

    for column in CATEGORICAL_COLUMNS + [TARGET_COLUMN]:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])
        encoders[column] = encoder

    print("4. Label-encoded categorical columns and the target.")
    return data, encoders


def scale_and_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """Standardize numerical features, then make an 80/20 stratified split."""
    feature_names = [column for column in data.columns if column != TARGET_COLUMN]
    scaler = StandardScaler()
    data = data.copy()
    data[NUMERICAL_COLUMNS] = scaler.fit_transform(data[NUMERICAL_COLUMNS])

    X = data[feature_names].to_numpy()
    y = data[TARGET_COLUMN].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("5. Scaled numerical columns and made an 80/20 stratified train/test split.")
    return X_train, X_test, y_train, y_test, scaler, feature_names


def save_outputs(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    encoders: dict[str, LabelEncoder],
    feature_names: list[str],
) -> None:
    """Save arrays and preprocessing artifacts needed by later application steps."""
    PROCESSED_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(encoders, ARTIFACTS_DIR / "encoders.pkl")
    joblib.dump(feature_names, ARTIFACTS_DIR / "feature_names.pkl")

    print("6. Saved arrays to 'processed/' and artifacts to 'artifacts/'.")
    print("   X_train:", X_train.shape)
    print("   X_test: ", X_test.shape)
    print("   y_train:", y_train.shape)
    print("   y_test: ", y_test.shape)


def main() -> None:
    """Run the complete CKD preprocessing pipeline."""
    data = load_data(DATA_PATH)
    data = clean_categories(data)
    data = impute_missing_values(data)
    data, encoders = encode_categories(data)
    X_train, X_test, y_train, y_test, scaler, feature_names = scale_and_split(data)
    save_outputs(X_train, X_test, y_train, y_test, scaler, encoders, feature_names)
    print("Preprocessing complete: all files were saved successfully.")


if __name__ == "__main__":
    main()
