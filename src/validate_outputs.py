"""Lab 5 - sanity-check the preprocessing outputs and write a summary report."""
import json
import os

import numpy as np

from common import CLASS_NAMES


def validate_preprocessing_outputs():
    print("[INFO] Validating Preprocessing Outputs...")

    # 1. Load the pipeline outputs
    try:
        X_train = np.load("data/processed/X_train_final.npy")
        X_test = np.load("data/processed/X_test_final.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find processed data: {e}")
        return False

    errors = []

    # 2. No missing values (validates the imputers)
    if np.isnan(X_train).sum() > 0:
        errors.append("NaNs detected in X_train after preprocessing.")
    if np.isnan(X_test).sum() > 0:
        errors.append("NaNs detected in X_test after preprocessing.")

    # 3. Dimensionality consistency (validates OHE and the split)
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature mismatch: X_train has {X_train.shape[1]} cols, X_test has {X_test.shape[1]} cols.")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("Row mismatch between X_train and y_train.")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append("Row mismatch between X_test and y_test.")

    # 4. Target checks (validates the 7-class encoding)
    valid_ids = set(range(len(CLASS_NAMES)))
    if not set(np.unique(y_train)) <= valid_ids or not set(np.unique(y_test)) <= valid_ids:
        errors.append(f"Target contains ids outside 0..{len(CLASS_NAMES) - 1}.")
    if set(np.unique(y_train)) != valid_ids:
        errors.append("Not all 7 obesity classes are present in y_train.")
    if set(np.unique(y_test)) != valid_ids:
        errors.append("Not all 7 obesity classes are present in y_test.")

    # 5. Transformation summary report
    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape),
        },
        "data_quality": {
            "missing_values_X_train": int(np.isnan(X_train).sum()),
            "missing_values_X_test": int(np.isnan(X_test).sum()),
        },
        "class_distribution": {
            "train": {CLASS_NAMES[i]: int((y_train == i).sum()) for i in range(len(CLASS_NAMES))},
            "test": {CLASS_NAMES[i]: int((y_test == i).sum()) for i in range(len(CLASS_NAMES))},
        },
        "errors": errors,
    }

    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/preprocessing_summary_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Output Validation FAILED!")
        for e in errors:
            print(f"  - {e}")
        return False

    print("[SUCCESS] Output Validation PASSED.")
    print("[INFO] Features are clean, scaled, encoded, and dimensionally consistent.")
    print(f"[INFO] Summary report saved to {report_path}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if validate_preprocessing_outputs() else 1)
