"""Shared settings and helpers so every script uses the same data, split and labels."""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
DATASET_NAME = "Obesity Level Estimation (Eating Habits and Physical Condition)"
EXPERIMENT_NAME = "Obesity_Level_Prediction"

TARGET = "NObeyesdad"
# Ordered from lightest to heaviest, so class ids 0..6 follow severity.
CLASS_NAMES = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

TEST_SIZE = 0.2
RANDOM_STATE = 42
RF_PARAMS = {"n_estimators": 200, "random_state": RANDOM_STATE}


def load_clean_data(path=RAW_DATA_PATH):
    """Load the raw CSV, drop exact duplicate rows and add the BMI feature.

    Returns (dataframe, number_of_duplicates_removed).
    """
    df = pd.read_csv(path)
    n_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    return df, n_duplicates


def get_features_and_target(df):
    """Split a cleaned dataframe into features X and integer-encoded target y."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str).str.strip().map(CLASS_TO_ID)
    if y.isna().any():
        unknown = sorted(df.loc[y.isna(), TARGET].astype(str).unique())
        raise ValueError(f"Unknown obesity classes in target column: {unknown}")
    return X, y.astype("int64")


def split_data(df):
    """Reproducible stratified 80/20 split -> X_train, X_test, y_train, y_test."""
    X, y = get_features_and_target(df)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def split_columns(X):
    """Return (numerical_columns, categorical_columns) of a feature dataframe."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return num_cols, cat_cols
