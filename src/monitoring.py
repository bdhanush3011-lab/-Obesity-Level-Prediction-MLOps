"""Simple numeric-feature PSI drift monitor; thresholds are configurable."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from common import RAW_DATA_PATH, load_clean_data

def psi(reference, current, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins+1)))
    if len(edges) < 3: return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, bins=edges)[0] / len(reference)
    c = np.histogram(current, bins=edges)[0] / len(current)
    r = np.clip(r, 1e-6, None); c = np.clip(c, 1e-6, None)
    return float(np.sum((c-r) * np.log(c/r)))

def run(input_path="data/incoming/new_obesity_data.csv"):
    ref, _ = load_clean_data()
    incoming = pd.read_csv(input_path)
    ref["BMI"] = ref.Weight / ref.Height**2
    incoming["BMI"] = incoming.Weight / incoming.Height**2
    numeric = ref.select_dtypes(include="number").columns.intersection(incoming.columns)
    scores = {col: psi(ref[col].dropna().to_numpy(), incoming[col].dropna().to_numpy())
              for col in numeric if len(incoming[col].dropna()) > 1}
    threshold = yaml.safe_load(Path("config.yaml").read_text())["monitoring"]["psi_warning_threshold"]
    report = {"status": "DRIFT_DETECTED" if any(v >= threshold for v in scores.values()) else "NO_SIGNIFICANT_DRIFT",
              "threshold": threshold, "psi": scores}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/drift_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run()
