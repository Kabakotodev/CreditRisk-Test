# app/prediction.py
import math
import os

import joblib

_MODEL = None
_MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
_MODEL_VERSION = "1.0.0"


def load_model(path: str = _MODEL_PATH):
    """Charge le modèle une seule fois."""
    global _MODEL
    if _MODEL is None and os.path.exists(path):
        _MODEL = joblib.load(path)
    return _MODEL


def get_model():
    return _MODEL


def is_ready() -> bool:
    return _MODEL is not None


def predict_proba(X) -> float:
    """Retourne la probabilité de défaut ∈ [0,1]."""
    model = get_model()
    if model is None:
        raise RuntimeError("Modèle non chargé")
    proba = float(model.predict_proba(X)[0, 1])
    if not math.isfinite(proba):
        raise RuntimeError("Probabilité non finie")
    return proba