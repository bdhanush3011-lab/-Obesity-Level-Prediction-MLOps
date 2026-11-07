"""Predict the obesity level of a new person (the notebook's final demo, as a script)."""
import os

import joblib
import numpy as np
import pandas as pd

from common import CLASS_NAMES


def load_model():
    path = "models/random_forest_model.pkl"
    if not os.path.exists(path):
        path = "models/random_forest_baseline.pkl"
    return joblib.load(path)


def transform(person):
    """Use the Lab 5 preprocessor if present, otherwise the Lab 1 scaler + encoder."""
    if os.path.exists("models/preprocessor.pkl"):
        return joblib.load("models/preprocessor.pkl").transform(person)

    scaler = joblib.load("models/scaler.pkl")
    ohe = joblib.load("models/ohe.pkl")
    num_cols = list(scaler.feature_names_in_)
    cat_cols = list(ohe.feature_names_in_)
    return np.hstack((scaler.transform(person[num_cols]), ohe.transform(person[cat_cols])))


def predict_level(person):
    person = person.copy()
    person["BMI"] = person["Weight"] / (person["Height"] ** 2)
    model = load_model()
    pred_id = int(model.predict(transform(person))[0])
    return CLASS_NAMES[pred_id]


if __name__ == "__main__":
    new_person = pd.DataFrame({
        "Age": [25.0],
        "Gender": ["Male"],
        "Height": [1.75],
        "Weight": [80.0],
        "CALC": ["Sometimes"],
        "FAVC": ["yes"],
        "FCVC": [2.0],
        "NCP": [3.0],
        "SCC": ["no"],
        "SMOKE": ["no"],
        "CH2O": [2.0],
        "family_history_with_overweight": ["yes"],
        "FAF": [1.0],
        "TUE": [1.0],
        "CAEC": ["Sometimes"],
        "MTRANS": ["Public_Transportation"],
    })
    print("New person:")
    print(new_person.T.to_string(header=False))
    print(f"\nPredicted Obesity Level: {predict_level(new_person)}")
