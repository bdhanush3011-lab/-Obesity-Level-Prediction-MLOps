"""Lab 5 - validate the raw obesity dataset against a strict schema before preprocessing."""
import os

import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column

from common import CLASS_NAMES, RAW_DATA_PATH

YES_NO = ["yes", "no"]
FREQUENCY = ["no", "Sometimes", "Frequently", "Always"]


def get_obesity_schema():
    """Defines the strict schema specification for the obesity dataset."""
    return pa.DataFrameSchema(
        {
            "Age": Column(pa.Float, Check.in_range(10, 100)),
            "Gender": Column(pa.String, Check.isin(["Male", "Female"])),
            "Height": Column(pa.Float, Check.in_range(1.0, 2.5)),          # metres
            "Weight": Column(pa.Float, Check.in_range(20.0, 300.0)),       # kilograms
            "CALC": Column(pa.String, Check.isin(FREQUENCY)),
            "FAVC": Column(pa.String, Check.isin(YES_NO)),
            "FCVC": Column(pa.Float, Check.in_range(1, 3)),
            "NCP": Column(pa.Float, Check.in_range(1, 4)),
            "SCC": Column(pa.String, Check.isin(YES_NO)),
            "SMOKE": Column(pa.String, Check.isin(YES_NO)),
            "CH2O": Column(pa.Float, Check.in_range(1, 3)),
            "family_history_with_overweight": Column(pa.String, Check.isin(YES_NO)),
            "FAF": Column(pa.Float, Check.in_range(0, 3)),
            "TUE": Column(pa.Float, Check.in_range(0, 2)),
            "CAEC": Column(pa.String, Check.isin(FREQUENCY)),
            "MTRANS": Column(
                pa.String,
                Check.isin(["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]),
            ),
            "NObeyesdad": Column(pa.String, Check.isin(CLASS_NAMES)),
        },
        strict=True,
    )


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates the input DataFrame against the defined schema."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_obesity_schema()

    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        print(f"[WARNING] {n_dupes} exact duplicate rows found (they are removed during preprocessing).")

    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True

    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")

        failures = err.failure_cases[["schema_context", "column", "check", "failure_case", "index"]]
        print(failures.to_string())

        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False


if __name__ == "__main__":
    if os.path.exists(RAW_DATA_PATH):
        raw_df = pd.read_csv(RAW_DATA_PATH)
        ok = validate_schema(raw_df, output_report_name="baseline_validation.csv")
        raise SystemExit(0 if ok else 1)
    else:
        print(f"[ERROR] Target file not found at: {RAW_DATA_PATH}")
        raise SystemExit(1)
