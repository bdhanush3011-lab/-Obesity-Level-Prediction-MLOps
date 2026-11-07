from fastapi import FastAPI, HTTPException
import pandas as pd
from api.schemas import ObesityFeatures, PredictionResponse
from api.model_loader import load_artifacts
from api.monitoring_logger import log_prediction
from src.common import CLASS_NAMES

app = FastAPI(title="Obesity Level Prediction API", version="1.0.0")

@app.get("/health")
def health():
    try:
        _, meta = load_artifacts()
        return {"status": "healthy", "model_loaded": True, "model_version": meta.get("version")}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}")

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ObesityFeatures):
    try:
        model, meta = load_artifacts()
        row = pd.DataFrame([payload.model_dump()])
        row["BMI"] = row["Weight"] / (row["Height"] ** 2)
        pred_id = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        confidence = float(max(probabilities))
        label = CLASS_NAMES[pred_id]
        log_prediction(payload.model_dump(), label, confidence)
        return {"predicted_obesity_level": label, "confidence": confidence,
                "model_version": str(meta.get("version", "unknown"))}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
