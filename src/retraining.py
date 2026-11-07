"""Conditional retraining: retrain a candidate when drift is detected."""
import json
from pathlib import Path
from monitoring import run as monitor
from train_production import run as train

def run():
    report_path = Path("reports/drift_report.json")
    if not report_path.exists():
        monitor()
    report = json.loads(report_path.read_text())
    if report["status"] == "DRIFT_DETECTED":
        print("[ACTION] Drift detected; training a new candidate.")
        train()
    else:
        print("[INFO] No significant drift; retraining skipped.")

if __name__ == "__main__":
    run()
