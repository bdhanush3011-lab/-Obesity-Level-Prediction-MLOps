"""Lab 5 - production preprocessing with a single scikit-learn ColumnTransformer pipeline."""
import json
import os

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CLASS_NAMES, DATASET_NAME, RAW_DATA_PATH, load_clean_data, split_columns, split_data


def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    # 1. Load, remove duplicates and engineer BMI
    df, n_duplicates = load_clean_data(RAW_DATA_PATH)
    print(f"[INFO] {len(df)} rows after removing {n_duplicates} duplicate rows")

    # 2. Train-Test Split (done before transformations to prevent leakage)
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Separate column types
    num_cols, cat_cols = split_columns(X_train)

    # 4. Modular scikit-learn pipelines
    # Numerical: fill missing values with the median, then scale
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical: fill missing values with the mode, then one-hot encode
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    # 5. Execute pipeline
    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)

    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    # 6. Save artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))

    # One preprocessor artifact instead of separate scaler/ohe
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    # 7. Save metadata
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

    print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")


if __name__ == "__main__":
    run_preprocessing()
