"""Explain one CKD TabNet prediction using its attention-based attributions."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib")))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain a TabNet test prediction.")
    parser.add_argument("--index", type=int, default=0, help="Test-sample index (default: 0)")
    return parser.parse_args()


def main() -> None:
    """Load artifacts, explain one test prediction, and save its bar chart."""
    args = parse_args()
    artifacts = Path("artifacts")
    results = Path("results")
    results.mkdir(exist_ok=True)

    X_test = np.load("processed/X_test.npy").astype(np.float32)
    feature_names = joblib.load(artifacts / "feature_names.pkl")
    encoders = joblib.load(artifacts / "encoders.pkl")
    if not 0 <= args.index < len(X_test):
        raise IndexError(f"--index must be between 0 and {len(X_test) - 1}.")

    model = TabNetClassifier(device_name="cpu")
    model.load_model(str(artifacts / "tabnet_model.zip"))
    print("1. Loaded TabNet model, test data, feature names, and label encoder.")

    attributions, _ = model.explain(X_test)
    sample = X_test[args.index : args.index + 1]
    probabilities = model.predict_proba(sample)[0]
    encoded_class = int(np.argmax(probabilities))
    label_encoder = encoders["classification"]
    predicted_class = label_encoder.inverse_transform([encoded_class])[0]
    confidence = float(probabilities[encoded_class])

    importance = attributions[args.index]
    order = np.argsort(importance)
    plt.figure(figsize=(8, 7))
    plt.barh(np.array(feature_names)[order], importance[order], color="teal")
    plt.xlabel("Attention-based attribution")
    plt.title(f"TabNet Explanation: Test Sample {args.index}")
    plt.tight_layout()
    output_path = results / "sample_explanation.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"2. Sample index: {args.index}")
    print(f"   Predicted class: {predicted_class}")
    print(f"   Confidence: {confidence:.4f}")
    print(f"3. Saved explanation plot to '{output_path}'.")


if __name__ == "__main__":
    main()
