"""Lab 1 - clean the data, split it, scale/encode the features and save the artifacts."""
import json
import os

import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CLASS_NAMES, DATASET_NAME, RAW_DATA_PATH, load_clean_data, split_columns, split_data


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    # 1. Load, remove duplicates and engineer BMI
    df, n_duplicates = load_clean_data(RAW_DATA_PATH)
    print(f"Loaded {RAW_DATA_PATH}: {len(df)} rows after removing {n_duplicates} duplicate rows")

    # 2. Train-Test Split (before any fitting, to prevent leakage)
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Separate column types
    num_cols, cat_cols = split_columns(X_train)

    # 4. Scale & Encode (fit on the training data only)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])

    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))

    # 5. Save artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(ohe, "models/ohe.pkl")

    # 6. Save metadata
    metadata = {
        "dataset_name": DATASET_NAME,
        "problem_type": "multiclass classification",
        "target": "NObeyesdad",
        "class_names": CLASS_NAMES,
        "rows_after_cleaning": int(len(df)),
        "duplicates_removed": n_duplicates,
        "engineered_features": ["BMI"],
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
