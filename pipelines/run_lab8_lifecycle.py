import json
from pathlib import Path
meta = Path("deployment/model_metadata.json")
if not meta.exists():
    raise SystemExit("No promoted model. Run Lab 6 first.")
data = json.loads(meta.read_text())
print("Current model stage:", data.get("stage"), "| version:", data.get("version"))
print("Lifecycle: registered -> staging (quality gate) -> production (promotion).")
