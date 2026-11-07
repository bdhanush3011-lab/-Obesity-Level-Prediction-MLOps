"""Simple filesystem model registry with immutable version folders."""
import json, shutil
from pathlib import Path
from datetime import datetime, timezone

def register_candidate(metrics):
    candidate = Path("models/production_candidate.pkl")
    if not candidate.exists():
        raise FileNotFoundError("Train a candidate first.")
    root = Path("registry"); root.mkdir(exist_ok=True)
    versions = [int(p.name[1:]) for p in root.glob("v[0-9]*") if p.name[1:].isdigit()]
    version = f"v{max(versions, default=0)+1}"
    dest = root/version; dest.mkdir()
    shutil.copy2(candidate, dest/"model.pkl")
    meta = {**metrics, "version": version, "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": "registered"}
    (dest/"metadata.json").write_text(json.dumps(meta, indent=2))
    return version, meta

def promote(version, meta):
    Path("deployment").mkdir(exist_ok=True)
    shutil.copy2(Path("registry")/version/"model.pkl", "deployment/model.pkl")
    prod = {**meta, "stage": "production"}
    Path("deployment/model_metadata.json").write_text(json.dumps(prod, indent=2))
    return prod
