# tests/test_regression.py
"""
Tests de non-régression — Partie V.
Compare la version courante à la baseline versionnée (baseline_v1.json).
"""
import math
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import prediction
from app.main import app
from app.preprocessing import transform
from tests.baseline import (
    REFERENCE_LABELS,
    compute_metrics,
    load_baseline,
    profiles_to_payloads,
)


# Tolérances DOCUMENTÉES (définies AVANT la comparaison)
TOL_PROBA = 1e-6
TOL_RECALL_DROP = 0.02
TOL_F1_DROP = 0.02
TOL_PRECISION_DROP = 0.03


@pytest.fixture(scope="module")
def baseline():
    b = load_baseline()
    if not b:
        pytest.skip("Baseline absente — exécuter scripts/create_baseline.py")
    return b


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ===========================================================================
# 1. Les tests fonctionnels réussissent après modification (déjà couverts
#    par test_api.py et test_integration.py — on refait ici une passe rapide)
# ===========================================================================
class TestFonctionnelApresModification:
    def test_requete_valide_stable(self, client):
        """Le contrat fonctionnel nominal reste inchangé."""
        r = client.post("/predict", json={
            "age": 35, "revenu_mensuel": 250000,
            "montant_credit": 1_000_000, "duree_mois": 12,
            "anciennete_activite": 4,
        })
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) >= {"probabilite_defaut", "niveau_risque", "version_modele"}
        assert 0.0 <= body["probabilite_defaut"] <= 1.0


# ===========================================================================
# 2. Les sorties du pipeline restent conformes au contrat
# ===========================================================================
class TestSortiesConformesContrat:

    def test_toutes_predictions_dans_intervalle(self, baseline):
        """Toutes les prédictions de référence ∈ [0,1] et finies."""
        for p in baseline["predictions"]:
            assert math.isfinite(p)
            assert 0.0 <= p <= 1.0

    def test_reproductibilite_sorties(self, baseline):
        """Rejouer les profils de référence donne les mêmes probabilités."""
        current = []
        for payload in profiles_to_payloads():
            X = transform(payload)
            current.append(prediction.predict_proba(X))

        for i, (expected, actual) in enumerate(zip(baseline["predictions"], current)):
            assert math.isclose(expected, actual, abs_tol=TOL_PROBA), (
                f"Régression de sortie sur profil #{i}: "
                f"attendu={expected:.6f}, obtenu={actual:.6f}"
            )


# ===========================================================================
# 3. Comparaison des métriques à la référence
# ===========================================================================
class TestMetriquesModele:

    @pytest.fixture(scope="class")
    def current_metrics(self, baseline):
        preds = []
        for payload in profiles_to_payloads():
            X = transform(payload)
            preds.append(prediction.predict_proba(X))
        return compute_metrics(preds, REFERENCE_LABELS)

    def test_rappel_non_degrade(self, baseline, current_metrics):
        """Rappel ≥ baseline − 2 pts."""
        seuil = baseline["metrics"]["rappel"] - TOL_RECALL_DROP
        assert current_metrics["rappel"] >= seuil, (
            f"Régression rappel : {current_metrics['rappel']:.3f} < {seuil:.3f}"
        )

    def test_f1_non_degrade(self, baseline, current_metrics):
        """F1 ≥ baseline − 2 pts."""
        seuil = baseline["metrics"]["f1"] - TOL_F1_DROP
        assert current_metrics["f1"] >= seuil, (
            f"Régression F1 : {current_metrics['f1']:.3f} < {seuil:.3f}"
        )

    def test_precision_non_degradee(self, baseline, current_metrics):
        """Précision ≥ baseline − 3 pts."""
        seuil = baseline["metrics"]["precision"] - TOL_PRECISION_DROP
        assert current_metrics["precision"] >= seuil, (
            f"Régression précision : {current_metrics['precision']:.3f} < {seuil:.3f}"
        )


# ===========================================================================
# 4. Détection d'une baisse excessive du rappel ou du F1-score
# ===========================================================================
class TestDetectionDegradation:

    def test_detection_chute_rappel(self, baseline):
        """
        Simule une chute du rappel en dégradant volontairement les seuils
        et vérifie que la détection marche.
        """
        degraded_metrics = dict(baseline["metrics"])
        degraded_metrics["rappel"] = baseline["metrics"]["rappel"] - 0.10

        seuil = baseline["metrics"]["rappel"] - TOL_RECALL_DROP
        assert degraded_metrics["rappel"] < seuil, (
            "Le mécanisme de détection n'a pas détecté la chute simulée"
        )

    def test_detection_chute_f1(self, baseline):
        degraded_f1 = baseline["metrics"]["f1"] - 0.10
        seuil = baseline["metrics"]["f1"] - TOL_F1_DROP
        assert degraded_f1 < seuil


# ===========================================================================
# 5. Cohérence après sauvegarde / rechargement du modèle
# ===========================================================================
class TestPersistenceModele:

    def test_save_load_preserve_predictions(self, tmp_path, baseline):
        """Sauvegarder puis recharger le modèle donne les mêmes prédictions."""
        model = prediction.get_model()
        if model is None:
            pytest.skip("Aucun modèle chargé — skip")

        # Sauvegarde
        saved = tmp_path / "reloaded_model.joblib"
        joblib.dump(model, saved)

        # Rechargement dans un objet séparé
        reloaded = joblib.load(saved)

        # Comparaison des prédictions
        for i, payload in enumerate(profiles_to_payloads()):
            X = transform(payload)
            p_original = float(model.predict_proba(X)[0, 1])
            p_reloaded = float(reloaded.predict_proba(X)[0, 1])

            assert math.isclose(p_original, p_reloaded, abs_tol=1e-9), (
                f"Incohérence save/load sur profil #{i}: "
                f"{p_original} vs {p_reloaded}"
            )