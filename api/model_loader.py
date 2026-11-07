import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "deployment" / "model.pkl"
META_PATH = ROOT / "deployment" / "model_metadata.json"

def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Production model missing. Run: python pipelines/run_lab6_registry.py")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text()) if META_PATH.exists() else {"version": "unversioned"}
    return model, metadata
