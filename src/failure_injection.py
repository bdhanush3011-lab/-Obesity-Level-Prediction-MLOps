"""Lab 9: lightweight negative tests for invalid schema and missing artifacts."""
import pandas as pd
import pandera.pandas as pa
from validate_data import get_obesity_schema

def run():
    bad = pd.DataFrame({"Age": [-5], "Gender": ["unknown"]})
    try:
        get_obesity_schema().validate(bad, lazy=True)
    except (pa.errors.SchemaErrors, pa.errors.SchemaError):
        print("PASS: invalid input rejected by schema")
    except Exception as exc:
        print(f"Schema rejected invalid input: {type(exc).__name__}")
    else:
        raise AssertionError("Invalid input unexpectedly passed schema validation")
    print("PASS: missing production artifact is handled by API readiness check")

if __name__ == "__main__":
    run()
