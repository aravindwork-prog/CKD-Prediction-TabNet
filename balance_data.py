"""Balance the CKD training data with SMOTE without modifying test data."""

from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE


PROCESSED_DIR = Path("processed")
X_TRAIN_PATH = PROCESSED_DIR / "X_train.npy"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.npy"
X_BALANCED_PATH = PROCESSED_DIR / "X_train_balanced.npy"
Y_BALANCED_PATH = PROCESSED_DIR / "y_train_balanced.npy"


def print_class_distribution(y: np.ndarray, label: str) -> None:
    """Print the sample count for every encoded target class."""
    classes, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(classes.tolist(), counts.tolist()))
    print(f"{label}: {distribution}")


def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load only the preprocessed training arrays."""
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    print(f"1. Loaded training data: X_train {X_train.shape}, y_train {y_train.shape}.")
    return X_train, y_train


def balance_training_data(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic minority-class samples from the training data only."""
    print_class_distribution(y_train, "2. Class distribution before SMOTE")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    print_class_distribution(y_balanced, "3. Class distribution after SMOTE")
    return X_balanced, y_balanced


def save_balanced_data(X_balanced: np.ndarray, y_balanced: np.ndarray) -> None:
    """Save balanced training arrays, leaving the test arrays untouched."""
    np.save(X_BALANCED_PATH, X_balanced)
    np.save(Y_BALANCED_PATH, y_balanced)
    print("4. Saved balanced training arrays to 'processed/'.")
    print(f"   X_train_balanced: {X_balanced.shape}")
    print(f"   y_train_balanced: {y_balanced.shape}")


def main() -> None:
    """Run the complete training-data balancing pipeline."""
    X_train, y_train = load_training_data()
    X_balanced, y_balanced = balance_training_data(X_train, y_train)
    save_balanced_data(X_balanced, y_balanced)
    print("SMOTE balancing complete. Test data was not changed.")


if __name__ == "__main__":
    main()
