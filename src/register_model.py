import json
from pathlib import Path
from registry import register_candidate, promote
from quality_gate import check

def run():
    metrics = json.loads(Path("artifacts/production_candidate_metrics.json").read_text())
    if not check(metrics):
        raise SystemExit("Candidate failed quality gate; not promoted.")
    version, meta = register_candidate(metrics)
    promote(version, meta)
    print(f"Registered and promoted {version} to Production.")

if __name__ == "__main__":
    run()
