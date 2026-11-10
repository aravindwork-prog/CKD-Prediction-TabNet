"""FastAPI endpoints for CKD TabNet predictions and prediction history."""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database
import model_service


class CKDInput(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    su: float
    rbc: str
    pc: str
    pcc: str
    ba: str
    bgr: float
    bu: float
    sc: float
    sod: float
    pot: float
    hemo: float
    pcv: float
    wc: float
    rc: float
    htn: str
    dm: str
    cad: str
    appet: str
    pe: str
    ane: str


app = FastAPI(title="CKD Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    database.init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def make_prediction(input_data: CKDInput) -> dict:
    raw_input = input_data.model_dump()
    result = model_service.predict(raw_input)
    database.log_prediction(raw_input, result["prediction"], result["confidence"], result["risk_level"])
    return result


@app.get("/history")
def history(limit: int = Query(default=10, ge=1, le=100)) -> list[dict]:
    return database.get_recent_predictions(limit)


@app.get("/feature-importance")
def feature_importance() -> list[dict]:
    return model_service.get_global_feature_importance()
