# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request

from app import prediction
from app.preprocessing import transform
from app.validation import CreditRequest

import math
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse



@asynccontextmanager
async def lifespan(_app: FastAPI):
    prediction.load_model()
    yield


app = FastAPI(title="Credit Risk API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    if not prediction.is_ready():
        raise HTTPException(status_code=503, detail="Modèle indisponible")
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {"version": prediction._MODEL_VERSION, "algo": "logistic_regression"}


@app.post("/predict")
def predict(payload: CreditRequest):
    if not prediction.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Modèle indisponible, veuillez réessayer plus tard.",
        )
    try:
        X = transform(payload.model_dump())
        proba = prediction.predict_proba(X)
    except Exception:
        # ⚠️ NE PAS exposer str(exc) : on renvoie un message générique
        raise HTTPException(status_code=500, detail="Erreur interne")

    niveau = "faible" if proba < 0.3 else "moyen" if proba < 0.7 else "eleve"
    return {
        "probabilite_defaut": proba,
        "niveau_risque": niveau,
        "version_modele": prediction._MODEL_VERSION,
    }



def _sanitize(value):
    """Rend les erreurs de validation JSON-sérialisables sans exposer de détails techniques."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]
    if isinstance(value, BaseException):
        return str(value)
    try:
        import json

        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": _sanitize(exc.errors())})