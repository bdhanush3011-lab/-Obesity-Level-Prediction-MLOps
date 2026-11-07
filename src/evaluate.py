"""Lab 2 - evaluate the baseline model and export a multiclass error analysis."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from common import CLASS_NAMES, load_clean_data, split_data

LABELS = list(range(len(CLASS_NAMES)))


def run_evaluation():
    print("Starting Model Evaluation...")

    # 1. Load test data and the trained model
    X_test_final = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")
    model = joblib.load("models/random_forest_baseline.pkl")

    # 2. Make predictions
    y_pred = model.predict(X_test_final)

    # 3. Calculate metrics (weighted averages because there are 7 classes)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy            : {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall    (weighted): {rec:.4f}")
    print(f"F1-Score  (weighted): {f1:.4f}")
    print("-------------------------------\n")

    print(classification_report(
        y_test, y_pred, labels=LABELS, target_names=CLASS_NAMES, zero_division=0
    ))

    # 4. Error analysis export - rebuild the same test rows from the cleaned data
    df, _ = load_clean_data()
    _, X_test_raw, _, y_test_check = split_data(df)
    if not np.array_equal(y_test_check.to_numpy(), y_test):
        raise RuntimeError(
            "Raw test rows do not line up with data/processed/y_test.npy. Re-run src/preprocess.py."
        )

    errors_df = X_test_raw.copy()
    errors_df["Actual_Level"] = [CLASS_NAMES[i] for i in y_test]
    errors_df["Predicted_Level"] = [CLASS_NAMES[i] for i in y_pred]
    errors_df["Level_Gap"] = np.abs(y_test - y_pred)

    misclassified = errors_df[errors_df["Actual_Level"] != errors_df["Predicted_Level"]]

    per_class = pd.DataFrame(
        classification_report(
            y_test, y_pred, labels=LABELS, target_names=CLASS_NAMES,
            zero_division=0, output_dict=True,
        )
    ).T

    os.makedirs("outputs", exist_ok=True)
    misclassified.to_csv("outputs/misclassified_samples.csv", index=False)
    per_class.to_csv("outputs/per_class_metrics.csv")

    print(f"Misclassified test samples: {len(misclassified)} of {len(errors_df)}")
    print("Evaluation complete and error analysis files saved!")


if __name__ == "__main__":
    run_evaluation()
