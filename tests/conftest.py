# tests/conftest.py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app.main import app
from app import prediction


VALID_PAYLOAD = {
    "age": 35,
    "revenu_mensuel": 250000,
    "montant_credit": 1_000_000,
    "duree_mois": 12,
    "anciennete_activite": 4,
}


@pytest.fixture(scope="session", autouse=True)
def _train_fake_model():
    """
    Entraîne un petit modèle logistique factice une seule fois par session
    et l'injecte dans app.prediction._MODEL.

    Ce modèle N'A AUCUNE VALEUR PRÉDICTIVE RÉELLE : il sert uniquement à
    valider le contrat technique de l'API (Parties II, III, IV, VI, VII).
    """
    X = np.array([
        [25, 100_000, 300_000,   6,  1],   # profil risqué
        [30, 200_000, 500_000,  12,  3],
        [35, 250_000, 1_000_000, 12, 4],
        [45, 500_000, 1_500_000, 24, 8],
        [55, 800_000, 2_000_000, 36, 15],  # profil sûr
    ], dtype=float)
    y = np.array([1, 1, 0, 0, 0])          # 1 = défaut

    model = LogisticRegression(max_iter=1000).fit(X, y)

    prediction._MODEL = model
    yield model
    prediction._MODEL = None


@pytest.fixture()
def client():
    """Client FastAPI synchrone, isolé par test."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def valid_payload():
    return dict(VALID_PAYLOAD)


@pytest.fixture()
def model_unavailable(monkeypatch):
    """
    Force le retour de get_model() à None pour simuler un modèle non chargé,
    sans toucher au modèle de session (restauré par monkeypatch).
    """
    monkeypatch.setattr(prediction, "get_model", lambda: None)
    monkeypatch.setattr(prediction, "is_ready", lambda: False)