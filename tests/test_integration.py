# tests/test_integration.py
"""
Tests d'intégration de bout en bout :
validation -> preprocessing -> modèle -> API.

Stratégie :
- Les tests E2E utilisent le VRAI pipeline (vraie validation, vrai
  preprocessing, vrai modèle de test injecté par la fixture de session).
- Les scénarios d'erreur mockent le minimum nécessaire pour forcer la
  défaillance, puis vérifient que l'API se protège.
"""
import math

import numpy as np
import pytest

from app import prediction


# ===========================================================================
# SCÉNARIO PRINCIPAL — E2E nominal
# ===========================================================================
class TestScenarioPrincipal:
    """6 étapes du scénario principal de la Partie III."""

    def test_etape1_requete_valide_acceptee(self, client, valid_payload):
        """Étape 1-2 : la requête valide franchit la validation et l'API répond 200."""
        r = client.post("/predict", json=valid_payload)
        assert r.status_code == 200, r.text

    def test_etape3_preprocessing_produit_matrice_valide(self, valid_payload):
        """Étape 3 : transform() produit une matrice (1, 5) en float."""
        from app.preprocessing import transform

        X = transform(valid_payload)

        assert X.shape == (1, 5)
        assert X.dtype == float
        assert np.all(np.isfinite(X))

    def test_etape4_prediction_executee_et_reproductible(self, client, valid_payload):
        """Étape 4 : deux appels identiques => même probabilité (à 1e-6)."""
        r1 = client.post("/predict", json=valid_payload)
        r2 = client.post("/predict", json=valid_payload)

        assert r1.status_code == 200
        assert r2.status_code == 200
        p1 = r1.json()["probabilite_defaut"]
        p2 = r2.json()["probabilite_defaut"]
        assert math.isclose(p1, p2, rel_tol=0.0, abs_tol=1e-6)

    def test_etape5_structure_de_reponse_conforme(self, client, valid_payload):
        """Étape 5 : contrat de réponse respecté (clés, types, bornes)."""
        r = client.post("/predict", json=valid_payload)
        body = r.json()

        assert set(body.keys()) >= {"probabilite_defaut", "niveau_risque", "version_modele"}
        assert isinstance(body["probabilite_defaut"], (int, float))
        assert math.isfinite(body["probabilite_defaut"])
        assert 0.0 <= body["probabilite_defaut"] <= 1.0
        assert body["niveau_risque"] in ("faible", "moyen", "eleve")
        assert isinstance(body["version_modele"], str) and body["version_modele"]

    def test_etape6_metadonnees_coherentes(self, client, valid_payload):
        """Étape 6 : /model-info renvoie la même version que /predict."""
        info = client.get("/model-info").json()
        pred = client.post("/predict", json=valid_payload).json()

        assert info["version"] == pred["version_modele"]


# ===========================================================================
# SCÉNARIOS COMPLÉMENTAIRES — gestion des erreurs
# ===========================================================================
class TestScenariosErreur:

    def test_int01_modele_non_chargeable(self, client, valid_payload, monkeypatch):
        """INT-01 : modèle absent en mémoire => 503 propre, sans trace."""
        monkeypatch.setattr(prediction, "get_model", lambda: None)
        monkeypatch.setattr(prediction, "is_ready", lambda: False)

        r = client.post("/predict", json=valid_payload)
        assert r.status_code == 503
        assert "detail" in r.json()
        _assert_no_leak(r.text)

    def test_int02_variable_absente(self, client, valid_payload, monkeypatch):
        """INT-02 : payload incomplet => 422, modèle jamais appelé."""
        called = {"n": 0}

        def _spy(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("Le modèle ne doit PAS être appelé")

        monkeypatch.setattr(prediction, "predict_proba", _spy)

        payload = dict(valid_payload)
        payload.pop("duree_mois")
        r = client.post("/predict", json=payload)

        assert r.status_code == 422
        assert called["n"] == 0  # le modèle n'a pas été sollicité

    def test_int03_erreur_preprocessing(self, client, valid_payload, monkeypatch):
        """INT-03 : preprocessing lève => 500 contrôlée, sans trace."""
        from app import main as main_module

        def _boom(_payload):
            raise ValueError("Erreur de préparation interne (simulée)")

        monkeypatch.setattr(main_module, "transform", _boom)

        r = client.post("/predict", json=valid_payload)
        assert r.status_code == 500
        assert "detail" in r.json()
        _assert_no_leak(r.text)

    def test_int04_dimensions_incorrectes(self, client, valid_payload, monkeypatch):
        """INT-04 : preprocessing renvoie mauvaise dimension => 500 contrôlée."""
        from app import main as main_module

        monkeypatch.setattr(
            main_module, "transform",
            lambda _payload: np.zeros((1, 3), dtype=float),  # 3 au lieu de 5
        )

        r = client.post("/predict", json=valid_payload)
        assert r.status_code == 500
        _assert_no_leak(r.text)

    def test_int05_erreur_pendant_prediction(self, client, valid_payload, monkeypatch):
        """INT-05 : le modèle lève pendant predict_proba => 500 contrôlée."""
        def _boom(_X):
            raise RuntimeError("Explosion interne du modèle (simulée)")

        monkeypatch.setattr(prediction, "predict_proba", _boom)

        r = client.post("/predict", json=valid_payload)
        assert r.status_code == 500
        _assert_no_leak(r.text)


# ===========================================================================
# Helpers
# ===========================================================================
def _assert_no_leak(text: str) -> None:
    """Vérifie qu'aucune information technique sensible ne fuit."""
    forbidden = ("Traceback", "File \"", ".py\", line", "site-packages", "sklearn/", "numpy/")
    for token in forbidden:
        assert token not in text, f"Fuite technique détectée : {token!r} dans {text[:200]!r}"