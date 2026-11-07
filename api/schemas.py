from pydantic import BaseModel, Field
from typing import Literal

class ObesityFeatures(BaseModel):
    Age: float = Field(ge=10, le=100)
    Gender: Literal["Male", "Female"]
    Height: float = Field(gt=1.0, le=2.5)
    Weight: float = Field(ge=20, le=300)
    CALC: Literal["no", "Sometimes", "Frequently", "Always"]
    FAVC: Literal["yes", "no"]
    FCVC: float = Field(ge=1, le=3)
    NCP: float = Field(ge=1, le=4)
    SCC: Literal["yes", "no"]
    SMOKE: Literal["yes", "no"]
    CH2O: float = Field(ge=1, le=3)
    family_history_with_overweight: Literal["yes", "no"]
    FAF: float = Field(ge=0, le=3)
    TUE: float = Field(ge=0, le=2)
    CAEC: Literal["no", "Sometimes", "Frequently", "Always"]
    MTRANS: Literal["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]

class PredictionResponse(BaseModel):
    predicted_obesity_level: str
    confidence: float
    model_version: str
