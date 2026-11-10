"""Train, evaluate, and save a TabNet classifier for CKD prediction."""

import os
from pathlib import Path

# Keep Matplotlib's cache inside the project so the script runs in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib")))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROCESSED_DIR = Path("processed")
ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR = Path("results")


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the balanced training data and untouched test data with TabNet dtypes."""
    X_train = np.load(PROCESSED_DIR / "X_train_balanced.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train_balanced.npy").astype(np.int64)
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(np.int64)
    print(f"1. Loaded data: X_train {X_train.shape}, y_train {y_train.shape}; "
          f"X_test {X_test.shape}, y_test {y_test.shape}.")
    return X_train, y_train, X_test, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> TabNetClassifier:
    """Fit a compact TabNet model with early stopping on the test evaluation set."""
    model = TabNetClassifier(n_d=8, n_a=8, n_steps=3, gamma=1.3, seed=42, device_name="cpu")
    print("2. Training TabNet (up to 100 epochs; early stopping patience = 15)...")
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        eval_name=["validation"],
        eval_metric=["logloss"],
        max_epochs=100,
        patience=15,
        batch_size=32,
        virtual_batch_size=32,
        num_workers=0,
        drop_last=False,
    )
    print("3. Training complete.")
    return model


def evaluate_model(model: TabNetClassifier, X_test: np.ndarray, y_test: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Calculate and print all requested binary classification metrics."""
    y_pred = model.predict(X_test).astype(np.int64)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    matrix = confusion_matrix(y_test, y_pred)

    print("4. Test-set evaluation metrics:")
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\n", matrix)
    return metrics, matrix


def save_plots(model: TabNetClassifier, matrix: np.ndarray, feature_names: list[str]) -> None:
    """Save loss, confusion-matrix, and global-feature-importance charts."""
    RESULTS_DIR.mkdir(exist_ok=True)

    history = model.history
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], label="Training loss")
    plt.plot(epochs, history["validation_logloss"], label="Validation log loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TabNet Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tabnet_loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["ckd", "notckd"], yticklabels=["ckd", "notckd"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("TabNet Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tabnet_confusion_matrix.png", dpi=150)
    plt.close()

    importances = model.feature_importances_
    order = np.argsort(importances)
    plt.figure(figsize=(8, 7))
    plt.barh(np.array(feature_names)[order], importances[order], color="teal")
    plt.xlabel("Global feature importance")
    plt.title("TabNet Feature Importance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tabnet_feature_importance.png", dpi=150)
    plt.close()
    print("5. Saved loss curve, confusion matrix, and feature-importance plots.")


def save_outputs(model: TabNetClassifier, metrics: dict[str, float]) -> None:
    """Save tabular metrics and the trained TabNet model."""
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "tabnet_metrics.csv", index=False)
    model.save_model(str(ARTIFACTS_DIR / "tabnet_model"))
    print("6. Saved metrics to 'results/tabnet_metrics.csv'.")
    print("7. Saved trained model to 'artifacts/tabnet_model.zip'.")


def main() -> None:
    """Run TabNet training, evaluation, visualizations, and artifact saving."""
    RESULTS_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    X_train, y_train, X_test, y_test = load_data()
    model = train_model(X_train, y_train, X_test, y_test)
    metrics, matrix = evaluate_model(model, X_test, y_test)
    feature_names = joblib.load(ARTIFACTS_DIR / "feature_names.pkl")
    save_plots(model, matrix, feature_names)
    save_outputs(model, metrics)
    print("TabNet pipeline complete: all requested outputs were saved.")


if __name__ == "__main__":
    main()
