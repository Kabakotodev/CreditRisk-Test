# tests/test_robustness.py
"""
Tests de robustesse — Partie IV.
Couvre : valeurs manquantes, non numériques, infinies/NaN, bornes,
valeurs très éloignées des données d'entraînement, schéma incorrect.

Principe : distinguer INVALIDE (4xx) de INHABITUEL MAIS VALIDE (200).
"""
import math
from xmlrpc import client

import pytest


BASE = {
    "age": 35,
    "revenu_mensuel": 250_000,
    "montant_credit": 1_000_000,
    "duree_mois": 12,
    "anciennete_activite": 4,
}


def _post(client, payload):
    return client.post("/predict", json=payload)


def _assert_no_leak(text: str) -> None:
    for token in ("Traceback", "File \"", ".py\", line", "site-packages"):
        assert token not in text, f"Fuite: {token!r} dans {text[:200]!r}"


def _assert_proba_valide(body: dict) -> None:
    p = body["probabilite_defaut"]
    assert isinstance(p, (int, float))
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


# ===========================================================================
# A. VALEURS MANQUANTES
# ===========================================================================
class TestValeursManquantes:

    @pytest.mark.parametrize("champ", list(BASE.keys()))
    def test_champ_absent(self, client, champ):
        """Chaque champ obligatoire manquant => 422 ciblant ce champ."""
        payload = dict(BASE)
        payload.pop(champ)
        r = _post(client, payload)
        assert r.status_code == 422
        locs = [e["loc"] for e in r.json()["detail"]]
        assert any(champ in loc for loc in locs), f"{champ} non signalé"

    @pytest.mark.parametrize("champ", list(BASE.keys()))
    def test_champ_null(self, client, champ):
        """null explicite sur un champ obligatoire => 422."""
        payload = dict(BASE, **{champ: None})
        r = _post(client, payload)
        assert r.status_code == 422
        _assert_no_leak(r.text)

    def test_payload_vide(self, client):
        """{} => 422 listant tous les champs."""
        r = _post(client, {})
        assert r.status_code == 422
        # toutes les clés obligatoires doivent être citées
        locs = {loc[-1] for e in r.json()["detail"] for loc in [e["loc"]]}
        assert set(BASE.keys()).issubset(locs)


# ===========================================================================
# B. VALEURS NON NUMÉRIQUES
# ===========================================================================
class TestNonNumeriques:

    @pytest.mark.parametrize("valeur", ["abc", "12abc", "", " ", "3.14.15", "None", "True", "False", "null"])
    def test_revenu_texte(self, client, valeur):
        payload = dict(BASE, revenu_mensuel=valeur)
        r = _post(client, payload)
        assert r.status_code == 422, f"'{valeur}' aurait dû être rejeté"
        _assert_no_leak(r.text)

    @pytest.mark.parametrize("valeur", [True, False])
    def test_booleen_pour_numerique(self, client, valeur):
        """
        Python convertit bool en int. On VÉRIFIE le comportement réel :
        - si Pydantic accepte (True=1, False=0) → 200 (comportement permissif à documenter)
        - si Pydantic rejette (strict) → 422
        Dans les deux cas, aucune 5xx et pas de fuite.
        """
        payload = dict(BASE, age=valeur)
        r = _post(client, payload)
        assert r.status_code in (200, 422)
        _assert_no_leak(r.text)
        if r.status_code == 200:
            _assert_proba_valide(r.json())

    @pytest.mark.parametrize("valeur", [[1, 2], {"x": 1}, "[]", "{}"])
    def test_structure_non_scalaire(self, client, valeur):
        payload = dict(BASE, montant_credit=valeur)
        r = _post(client, payload)
        assert r.status_code == 422
        _assert_no_leak(r.text)


# ===========================================================================
# C. VALEURS INFINIES / NON FINIES
# ===========================================================================
class TestInfiniNaN:

    def test_infini_positif(self, client):
        """1e309 devient inf en Python → doit être rejeté (plafond métier)."""
        payload = dict(BASE, revenu_mensuel=1e309)
        r = _post(client, payload)
        assert r.status_code == 422
        _assert_no_leak(r.text)

    def test_infini_negatif(self, client):
        payload = dict(BASE, revenu_mensuel=-1e309)
        r = _post(client, payload)
        assert r.status_code == 422
        _assert_no_leak(r.text)

    def test_nan_texte(self, client):
        """'NaN' sous forme de chaîne => 422 (non numérique)."""
        payload = dict(BASE, revenu_mensuel="NaN")
        r = _post(client, payload)
        assert r.status_code == 422

    def test_nan_json_litteral(self, client):
        """JSON n'autorise pas NaN nativement : on envoie du brut pour tester le parseur."""
        r = client.post(
            "/predict",
            content=(
                '{"age":35,"revenu_mensuel":NaN,"montant_credit":1000000,'
                '"duree_mois":12,"anciennete_activite":4}'
            ),
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code in (400, 422)
        _assert_no_leak(r.text)

    # def test_proba_toujours_finie(self, client):
    #     """Invariant : quelle que soit l'entrée valide, la sortie est finie."""
    #     for payload in [
    #         BASE,
    #         dict(BASE, age=18),
    #         dict(BASE, age=100),
    #         dict(BASE, revenu_mensuel=1),
    #         dict(BASE, montant_credit=5_000_000),
    #         dict(BASE, duree_mois=360),
    #     ]:
    #         r = _post(client, payload)
    #         assert r.status_code == 200
    #         _assert_proba_valide(r.json())

####
    def test_proba_toujours_finie(self, client):
        """Invariant : quelle que soit l'entrée valide, la sortie est finie."""
        for payload in [
            BASE,
            dict(BASE, age=18, anciennete_activite=0),          # ✅ incohérence évitée
            dict(BASE, age=100),
            dict(BASE, revenu_mensuel=1),
            dict(BASE, montant_credit=5_000_000),
            dict(BASE, duree_mois=360),
        ]:
            r = _post(client, payload)
            assert r.status_code == 200, f"Échec sur {payload}: {r.text}"
            _assert_proba_valide(r.json())

# ===========================================================================
# D. VALEURS AUX BORNES DES INTERVALLES
# ===========================================================================
class TestBornes:

    # @pytest.mark.parametrize("age,attendu", [
    #     (18, 200), (100, 200),
    #     (17, 422), (101, 422),
    # ])
    # def test_age_bornes(self, client, age, attendu):
    #     payload = dict(BASE, age=age)
    #     r = _post(client, payload)
    #     assert r.status_code == attendu
    #     if attendu == 200:
    #         _assert_proba_valide(r.json())

    @pytest.mark.parametrize("age,anciennete,attendu", [
    (18,   0, 200),   # borne basse + ancienneté min
    (100, 60, 200),   # borne haute + ancienneté max
    (17,   0, 422),   # sous la borne
    (101,  0, 422),   # au-dessus de la borne
])
    def test_age_bornes(self, client, age, anciennete, attendu):
        payload = dict(BASE, age=age, anciennete_activite=anciennete)
        r = _post(client, payload)
        assert r.status_code == attendu, f"age={age} anciennete={anciennete}: {r.text}"
        if attendu == 200:
            _assert_proba_valide(r.json())
####

    @pytest.mark.parametrize("duree,attendu", [
        (1, 200), (360, 200),
        (0, 422), (361, 422),
    ])
    def test_duree_bornes(self, client, duree, attendu):
        payload = dict(BASE, duree_mois=duree)
        r = _post(client, payload)
        assert r.status_code == attendu

    @pytest.mark.parametrize("montant,attendu", [
        (1, 200), (5_000_000, 200),
        (0, 422), (5_000_001, 422),
    ])
    def test_montant_bornes(self, client, montant, attendu):
        payload = dict(BASE, montant_credit=montant)
        r = _post(client, payload)
        assert r.status_code == attendu

    def test_anciennete_incoherente(self, client):
        """Ancienneté > âge - 18 => 422 (validation croisée)."""
        payload = dict(BASE, age=25, anciennete_activite=20)
        r = _post(client, payload)
        assert r.status_code == 422
        _assert_no_leak(r.text)


# ===========================================================================
# E. ENTRÉES INHABITUELLES MAIS VALIDES (proches des bornes / hors distribution)
# ===========================================================================
class TestHorsDistribution:
    """
    Entrées très éloignées des données d'entraînement mais DANS les bornes.
    Comportement attendu : 200 (accepté) + probabilité finie.
    Alerte côté métier possible (à documenter comme recommandation).
    """

    def test_profil_jeune_tres_risque(self, client):
        payload = dict(BASE, age=18, revenu_mensuel=1, montant_credit=5_000_000,
                       duree_mois=360, anciennete_activite=0)
        r = _post(client, payload)
        assert r.status_code == 200
        _assert_proba_valide(r.json())

    def test_profil_senior_tres_sur(self, client):
        payload = dict(BASE, age=99, revenu_mensuel=10_000_000, montant_credit=1,
                       duree_mois=1, anciennete_activite=60)
        r = _post(client, payload)
        assert r.status_code == 200
        _assert_proba_valide(r.json())

    def test_proba_stable_sur_entrees_inhabituelles(self, client):
        """Deux appels identiques sur profil atypique => même probabilité."""
        payload = dict(BASE, age=18, revenu_mensuel=1, montant_credit=5_000_000,
                       duree_mois=360, anciennete_activite=0)
        r1 = _post(client, payload).json()["probabilite_defaut"]
        r2 = _post(client, payload).json()["probabilite_defaut"]
        assert math.isclose(r1, r2, abs_tol=1e-6)


# ===========================================================================
# F. SCHÉMA INCORRECT
# ===========================================================================
class TestSchemaIncorrect:

    def test_payload_liste_au_lieu_objet(self, client):
        r = client.post("/predict", json=[1, 2, 3])
        assert r.status_code == 422
        _assert_no_leak(r.text)

    def test_payload_null(self, client):
        r = client.post("/predict", json=None)
        assert r.status_code == 422
        _assert_no_leak(r.text)

    def test_payload_scalaire(self, client):
        r = client.post("/predict", json=42)
        assert r.status_code == 422
        _assert_no_leak(r.text)

    def test_content_type_incorrect(self, client):
        r = client.post(
            "/predict",
            content="age=35&revenu_mensuel=250000",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert r.status_code in (400, 415, 422)
        _assert_no_leak(r.text)

    def test_json_tronque(self, client):
        r = client.post(
            "/predict",
            content='{"age":35,"revenu_mensuel":',
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code in (400, 422)
        _assert_no_leak(r.text)

    def test_json_double_encodage(self, client):
        """Payload JSON encodé comme une chaîne → 422."""
        payload = '{"age":35,"revenu_mensuel":250000,"montant_credit":1000000,"duree_mois":12,"anciennete_activite":4}'
        r = client.post("/predict", json=payload)
        assert r.status_code == 422
        _assert_no_leak(r.text)

    