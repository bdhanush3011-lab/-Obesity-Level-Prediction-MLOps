# Obesity Level Prediction – End-to-End MLOps

Multiclass classification for the seven `NObeyesdad` labels using the original Obesity
Level Estimation dataset. This upgrade preserves the supplied notebook, paper, dataset,
Lab 1–5 scripts, model experimentation, and existing outputs, and adds Labs 6–13.

## Project scope and caveats

- The supplied dataset contains 2,111 records and 17 original columns; the project removes
  exact duplicates and engineers BMI as in the original implementation.
- The target has seven obesity-level classes.
- BMI is mathematically related to the target definition. High test scores should not be
  interpreted as proof of broad real-world generalization. The dataset documentation also
  describes a substantial synthetic-data component.
- This project is an educational MLOps demonstration, not a medical diagnostic system.
- MLflow tracking and the original Lab 1–5 implementation remain available. The added
  registry is a simple local filesystem registry for teaching; it is not a remote registry.
- Monitoring uses a basic PSI calculation on numeric feature distributions. It is a
  demonstrator, not a complete production monitoring platform.

## Setup (Windows PowerShell)

Run commands from the `MLOPS-PROJECT` root:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-api.txt
```

If PowerShell activation is blocked, use Command Prompt instead:
`.venv\Scripts\activate.bat`.

## Labs

| Lab | Topic | Command |
|---|---|---|
| 1–3 | Ingestion/preprocessing, baseline training and evaluation | `python pipelines/run_lab3_baseline.py` |
| 4 | MLflow tracking and reproducibility | `python pipelines/run_lab4_tracking.py` |
| 5 | Schema validation and production preprocessing | `python pipelines/run_lab5_pipeline.py` |
| 6 | Candidate training, quality gate, registry and promotion | `python pipelines/run_lab6_registry.py` |
| 7 | Re-run quality gate report | `python pipelines/run_lab7_quality_gate.py` |
| 8 | Inspect model lifecycle / production version | `python pipelines/run_lab8_lifecycle.py` |
| 9 | Failure-handling checks | `python pipelines/run_lab9_failure.py` |
| 10 | API launch instructions | `python pipelines/run_lab10_api.py` |
| 11 | Docker | `docker build -t obesity-level-api:1.0 .` |
| 12 | Kubernetes | `kubectl apply -f k8s/` |
| 13 | Drift monitoring | `python pipelines/run_lab13_monitoring.py` |

## Production model and API

First run Lab 6 to create `deployment/model.pkl` and its metadata. Then:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`. `GET /health` checks that the production artifact can
be loaded. `POST /predict` accepts the 16 original feature columns (BMI is engineered by
the service) and returns the predicted class and model confidence.

Example request body:

```json
{
  "Age": 25, "Gender": "Male", "Height": 1.75, "Weight": 80,
  "CALC": "Sometimes", "FAVC": "yes", "FCVC": 2, "NCP": 3,
  "SCC": "no", "SMOKE": "no", "CH2O": 2,
  "family_history_with_overweight": "yes", "FAF": 1, "TUE": 1,
  "CAEC": "Sometimes", "MTRANS": "Public_Transportation"
}
```

Predictions are appended to `logs/predictions.jsonl`.

## Monitoring and retraining

Place a new, unlabeled batch CSV at `data/incoming/new_obesity_data.csv`. It should contain
the 16 feature columns used by the model; the monitor derives BMI. Run:

```powershell
python pipelines/run_lab13_monitoring.py
python pipelines/run_retraining.py
```

Retraining is triggered only when at least one numeric feature's PSI reaches the configured
threshold in `config.yaml`. Since this dataset has no natural time-based incoming feed,
create a genuinely new batch before interpreting drift results. Do not treat the training
CSV copied unchanged as a meaningful production batch.

## Docker and Kubernetes

After Lab 6, build the image. The model artifact must be present in the build context.
For local Docker Compose: `docker compose up --build`.
For Kubernetes, make the image available to your cluster first, then apply the manifests.
The manifests are templates and have not been validated against your specific cluster.

## Tests and CI

```powershell
pip install pytest httpx pyyaml
pytest -q
```

GitHub Actions runs compile checks and tests on pushes and pull requests.

## Original project results

The original README reports reference-run accuracy and weighted F1 around 0.9856 for its
Random Forest. These are historical reference figures from the supplied project, not a
claim that this new deployment pipeline has been executed in every environment. The
new production pipeline writes its own metrics to `artifacts/production_candidate_metrics.json`.

## Troubleshooting

- Run all commands from the project root.
- If PowerShell blocks activation, use the temporary Process-scope execution policy shown
  above or activate from Command Prompt.
- If API `/health` returns 503, run Lab 6 and check `deployment/`.
- If MLflow UI is needed: `mlflow ui` then open `http://127.0.0.1:5000`.
