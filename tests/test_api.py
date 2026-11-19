# tests/test_api.py
"""
Tests fonctionnels de l'API de prédiction du risque de crédit.
Couvre les 10 scénarios obligatoires de la Partie II.
"""
import math

import pytest


# ---------------------------------------------------------------------------
# 1. Requête valide → 200 + probabilité ∈ [0,1]
# ---------------------------------------------------------------------------
def test_requete_valide(client, valid_payload):
    r = client.post("/predict", json=valid_payload)

    assert r.status_code == 200, r.text
    body = r.json()

    assert "probabilite_defaut" in body
    proba = body["probabilite_defaut"]
    assert isinstance(proba, (int, float))
    assert math.isfinite(proba)
    assert 0.0 <= proba <= 1.0


# ---------------------------------------------------------------------------
# 2. Champ obligatoire manquant → 422
# ---------------------------------------------------------------------------
def test_champ_obligatoire_manquant(client, valid_payload):
    payload = dict(valid_payload)
    payload.pop("age")

    r = client.post("/predict", json=payload)

    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any("age" in str(err.get("loc", [])) for err in detail)


# ---------------------------------------------------------------------------
# 3. Âge négatif → 422
# ---------------------------------------------------------------------------
def test_age_negatif(client, valid_payload):
    payload = dict(valid_payload, age=-5)

    r = client.post("/predict", json=payload)

    assert r.status_code == 422
    assert "age" in r.text


# ---------------------------------------------------------------------------
# 4. Revenu sous forme de texte → 422
# ---------------------------------------------------------------------------
def test_revenu_texte(client, valid_payload):
    payload = dict(valid_payload, revenu_mensuel="beaucoup")

    r = client.post("/predict", json=payload)

    assert r.status_code == 422
    assert "revenu_mensuel" in r.text


# ---------------------------------------------------------------------------
# 5. Montant nul ou négatif → 422
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("montant", [0, -1, -1_000_000])
def test_montant_nul_ou_negatif(client, valid_payload, montant):
    payload = dict(valid_payload, montant_credit=montant)

    r = client.post("/predict", json=payload)

    assert r.status_code == 422
    assert "montant_credit" in r.text


# ---------------------------------------------------------------------------
# 6. Durée non entière → 422
# ---------------------------------------------------------------------------
def test_duree_non_entiere(client, valid_payload):
    payload = dict(valid_payload, duree_mois=12.5)

    r = client.post("/predict", json=payload)

    assert r.status_code == 422
    assert "duree_mois" in r.text


# ---------------------------------------------------------------------------
# 7. Champs supplémentaires non autorisés → 422 (extra="forbid")
# ---------------------------------------------------------------------------
def test_champ_supplementaire_interdit(client, valid_payload):
    payload = dict(valid_payload, sexe="M")

    r = client.post("/predict", json=payload)

    assert r.status_code == 422
    assert "sexe" in r.text


# ---------------------------------------------------------------------------
# 8. Valeurs extrêmement élevées → 422 (plafonds métier)
# ---------------------------------------------------------------------------
def test_valeurs_extremement_elevees(client, valid_payload):
    payload = dict(
        valid_payload,
        revenu_mensuel=10**12,
        montant_credit=10**12,
        duree_mois=10_000,
    )

    r = client.post("/predict", json=payload)

    assert r.status_code == 422
    # au moins un des champs doit être signalé
    assert any(k in r.text for k in ("revenu_mensuel", "montant_credit", "duree_mois"))


# ---------------------------------------------------------------------------
# 9. Modèle indisponible → 503, pas de stack trace
# ---------------------------------------------------------------------------
def test_modele_indisponible(client, valid_payload, model_unavailable):
    r = client.post("/predict", json=valid_payload)

    assert r.status_code == 503
    body = r.json()
    assert "detail" in body
    # aucune fuite technique
    assert "Traceback" not in r.text
    assert "File \"" not in r.text


# ---------------------------------------------------------------------------
# 10. Requête mal formée (JSON invalide) → 4xx contrôlée
# ---------------------------------------------------------------------------
def test_requete_mal_formee(client):
    r = client.post(
        "/predict",
        content="{ceci n'est pas du JSON",
        headers={"Content-Type": "application/json"},
    )

    assert r.status_code in (400, 422)
    assert "Traceback" not in r.text
    assert "File \"" not in r.text


# ---------------------------------------------------------------------------
# Bonus : /health et /model-info (utiles pour le rapport)
# ---------------------------------------------------------------------------
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") in ("ok", "healthy")


def test_health_modele_indisponible(client, model_unavailable):
    r = client.get("/health")
    assert r.status_code == 503


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body and body["version"]