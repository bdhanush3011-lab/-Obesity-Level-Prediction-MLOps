"""Lab 4 - train several models with MLflow tracking (params, metrics, plots, metadata, model)."""
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

from common import CLASS_NAMES, EXPERIMENT_NAME, RANDOM_STATE, RF_PARAMS

LABELS = list(range(len(CLASS_NAMES)))


def load_arrays():
    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_test, y_pred, run_name, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=LABELS, display_labels=CLASS_NAMES,
        ax=ax, cmap="Blues", xticks_rotation=45,
    )
    ax.set_title(f"Confusion Matrix - {run_name}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_one_vs_rest(y_test, y_prob, run_name, path):
    y_bin = label_binarize(y_test, classes=LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(CLASS_NAMES):
        RocCurveDisplay.from_predictions(y_bin[:, i], y_prob[:, i], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_title(f"ROC Curves (one-vs-rest) - {run_name}")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def train_and_track(run_name, model_class, params, save_local_as=None):
    print(f"\n--- Starting MLflow Run: {run_name} ---")

    X_train, X_test, y_train, y_test = load_arrays()
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        # 1. Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", model_class.__name__)

        # 2. Train
        model = model_class(**params)
        model.fit(X_train, y_train)

        # 3. Evaluate (weighted averages / one-vs-rest AUC for 7 classes)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "roc_auc_ovr": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro", labels=LABELS),
        }
        mlflow.log_metrics(metrics)
        print(f"Metrics logged: accuracy = {metrics['accuracy']:.4f} | "
              f"F1 (weighted) = {metrics['f1_weighted']:.4f} | ROC-AUC (OvR) = {metrics['roc_auc_ovr']:.4f}")

        # 4. Diagnostic plots
        os.makedirs("artifacts", exist_ok=True)
        cm_path = f"artifacts/{run_name}_confusion_matrix.png"
        roc_path = f"artifacts/{run_name}_roc_curve.png"
        plot_confusion_matrix(y_test, y_pred, run_name, cm_path)
        plot_roc_one_vs_rest(y_test, y_prob, run_name, roc_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # 5. Preprocessing metadata for lineage
        if os.path.exists("data/processed/dataset_metadata.json"):
            mlflow.log_artifact("data/processed/dataset_metadata.json", artifact_path="metadata")

        # 6. Log the model
        mlflow.sklearn.log_model(sk_model=model, name="model")

        # 7. Optional local backup
        if save_local_as:
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, save_local_as)
        print(f"Run '{run_name}' successfully tracked!")

    return metrics


def tune_random_forest():
    """Same grid as the notebook: GridSearchCV with 5-fold CV on the training data."""
    X_train, _, y_train, _ = load_arrays()
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        cv=5, scoring="accuracy", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"GridSearchCV best params: {grid.best_params_} (CV accuracy {grid.best_score_:.4f})")
    return {**grid.best_params_, "random_state": RANDOM_STATE}


if __name__ == "__main__":
    # Run 1 - baseline Random Forest (also saved locally as models/random_forest_model.pkl)
    train_and_track("RandomForest", RandomForestClassifier, RF_PARAMS,
                    save_local_as="models/random_forest_model.pkl")

    # Run 2 - tuned Random Forest (GridSearchCV, as in the notebook)
    train_and_track("RandomForest_Tuned", RandomForestClassifier, tune_random_forest())

    # Runs 3-5 - the other baselines from the notebook, for experiment comparison
    train_and_track("DecisionTree", DecisionTreeClassifier, {"random_state": RANDOM_STATE})
    train_and_track("LogisticRegression", LogisticRegression, {"max_iter": 2000})
    train_and_track("KNN", KNeighborsClassifier, {"n_neighbors": 5})
