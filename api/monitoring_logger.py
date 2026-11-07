import csv, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "predictions.jsonl"

def log_prediction(features, prediction, confidence):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(),
           "features": features, "prediction": prediction, "confidence": confidence}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
