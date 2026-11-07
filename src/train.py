"""Lab 2 - train the baseline Random Forest on the processed training arrays."""
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from common import RF_PARAMS


def run_training():
    print("Starting Model Training...")

    # 1. Load processed training data
    X_train_final = np.load("data/processed/X_train_final.npy")
    y_train = np.load("data/processed/y_train.npy")

    # 2. Initialize and train the model
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train_final, y_train)

    # 3. Save the trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_baseline.pkl")
    print("Model training complete and saved to disk!")


if __name__ == "__main__":
    run_training()
