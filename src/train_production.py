"""Train one deployable end-to-end sklearn pipeline on raw feature columns."""
import json
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from common import RAW_DATA_PATH, TARGET, CLASS_NAMES, CLASS_TO_ID, RANDOM_STATE, load_clean_data, split_data, split_columns

def run():
    df, duplicates = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)
    num, cat = split_columns(X_train)
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    model = Pipeline([("preprocessor", prep),
                      ("classifier", RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE,
                                                            class_weight="balanced_subsample"))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"accuracy": float(accuracy_score(y_test, pred)),
               "f1_macro": float(f1_score(y_test, pred, average="macro"))}
    Path("models").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(model, "models/production_candidate.pkl")
    Path("artifacts/production_candidate_metrics.json").write_text(json.dumps({
        **metrics, "rows": len(df), "duplicates_removed": duplicates,
        "classes": CLASS_NAMES, "random_state": RANDOM_STATE
    }, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    run()
