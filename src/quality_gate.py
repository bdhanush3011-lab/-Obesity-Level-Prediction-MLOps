import json, sys
from pathlib import Path
import yaml

def check(metrics):
    cfg = yaml.safe_load(Path("config.yaml").read_text())["quality_gate"]
    failures = {k: {"actual": metrics.get(k), "required": v}
                for k, v in cfg.items() if metrics.get(k, 0) < v}
    report = {"status": "PASSED" if not failures else "FAILED", "failures": failures, "metrics": metrics}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/quality_gate_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return not failures

if __name__ == "__main__":
    m = json.loads(Path("artifacts/production_candidate_metrics.json").read_text())
    sys.exit(0 if check(m) else 1)
