"""Lab 4 - train the same model twice with fixed parameters and confirm identical scores."""
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from common import RF_PARAMS


def run_deterministic_test():
    print("Running Reproducibility Validation...")

    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    params = dict(RF_PARAMS)

    def one_execution():
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return f1_score(y_test, pred, average="weighted"), accuracy_score(y_test, pred)

    f1_1, acc_1 = one_execution()
    f1_2, acc_2 = one_execution()  # simulated re-run

    is_reproducible = (f1_1 == f1_2) and (acc_1 == acc_2)

    report = {
        "test_name": "Pipeline Reproducibility Validation",
        "parameters": params,
        "execution_1_f1_weighted": f1_1,
        "execution_2_f1_weighted": f1_2,
        "execution_1_accuracy": acc_1,
        "execution_2_accuracy": acc_2,
        "is_strictly_reproducible": bool(is_reproducible),
        "status": "PASSED" if is_reproducible else "FAILED",
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/reproducibility_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"Execution 1 F1 (weighted): {f1_1:.6f}")
    print(f"Execution 2 F1 (weighted): {f1_2:.6f}")
    if is_reproducible:
        print("SUCCESS: Pipeline is 100% reproducible. Report saved.")
    else:
        print("FAILED: Pipeline is non-deterministic.")
        raise SystemExit(1)


if __name__ == "__main__":
    run_deterministic_test()
