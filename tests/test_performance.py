# tests/test_performance.py
"""
Tests de performance — Partie VI.
Mesure latence / débit / stabilité de l'API sous charge locale.
"""
import statistics
import time

import pytest
from fastapi.testclient import TestClient

from app import prediction
from app.main import app
from app.preprocessing import transform


PAYLOAD = {
    "age": 35,
    "revenu_mensuel": 250_000,
    "montant_credit": 1_000_000,
    "duree_mois": 12,
    "anciennete_activite": 4,
}

# Objectifs (identiques au rapport)
TARGET_P95_MS = 200.0
TARGET_MEAN_MS = 100.0
TARGET_INFERENCE_MS = 10.0
N_ITERATIONS = 100


def _percentile(values, p):
    """Percentile par interpolation linéaire (méthode numpy)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ===========================================================================
# 1. Temps de réponse de l'API /predict
# ===========================================================================
class TestLatenceAPI:

    def test_latence_predict(self, client):
        """Mesure min / moyenne / P95 / max sur N itérations."""
        latencies_ms = []
        # warm-up (3 appels)
        for _ in range(3):
            client.post("/predict", json=PAYLOAD)

        for _ in range(N_ITERATIONS):
            t0 = time.perf_counter()
            r = client.post("/predict", json=PAYLOAD)
            t1 = time.perf_counter()
            assert r.status_code == 200
            latencies_ms.append((t1 - t0) * 1000.0)

        stats = {
            "n": len(latencies_ms),
            "min_ms": min(latencies_ms),
            "mean_ms": statistics.mean(latencies_ms),
            "median_ms": statistics.median(latencies_ms),
            "p95_ms": _percentile(latencies_ms, 95),
            "max_ms": max(latencies_ms),
        }

        print("\n[LATENCE API]", stats)

        # Assertions sur les objectifs
        assert stats["mean_ms"] < TARGET_MEAN_MS, (
            f"Moyenne {stats['mean_ms']:.1f}ms ≥ cible {TARGET_MEAN_MS}ms"
        )
        assert stats["p95_ms"] < TARGET_P95_MS, (
            f"P95 {stats['p95_ms']:.1f}ms ≥ cible {TARGET_P95_MS}ms"
        )


# ===========================================================================
# 2. Latence d'inférence seule (sans passer par l'API)
# ===========================================================================
class TestLatenceInference:

    def test_latence_modele_seul(self):
        """Mesure le temps du modèle uniquement (préprocessing + predict_proba)."""
        if not prediction.is_ready():
            pytest.skip("Modèle non chargé")

        X = transform(PAYLOAD)
        # warm-up
        for _ in range(3):
            prediction.predict_proba(X)

        latencies_ms = []
        for _ in range(N_ITERATIONS):
            t0 = time.perf_counter()
            prediction.predict_proba(X)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        stats = {
            "n": len(latencies_ms),
            "mean_ms": statistics.mean(latencies_ms),
            "p95_ms": _percentile(latencies_ms, 95),
            "max_ms": max(latencies_ms),
        }

        print("\n[LATENCE INFERENCE]", stats)

        assert stats["p95_ms"] < TARGET_INFERENCE_MS, (
            f"P95 inférence {stats['p95_ms']:.2f}ms ≥ cible {TARGET_INFERENCE_MS}ms"
        )


# ===========================================================================
# 3. Débit en mono-thread (séquentiel)
# ===========================================================================
class TestDebit:

    def test_debit_sequentiel(self, client):
        """Estime le débit maximal soutenable en mono-thread."""
        n = 50
        # warm-up
        for _ in range(3):
            client.post("/predict", json=PAYLOAD)

        t0 = time.perf_counter()
        for _ in range(n):
            r = client.post("/predict", json=PAYLOAD)
            assert r.status_code == 200
        t1 = time.perf_counter()

        elapsed = t1 - t0
        rps = n / elapsed

        print(f"\n[DEBIT] {n} req en {elapsed:.3f}s → {rps:.1f} req/s (mono-thread)")

        # TestClient séquentiel ≈ 1 worker → on attend un débit correct
        # (le vrai débit multi-utilisateurs est mesuré avec Locust, cf. §5)
        assert rps > 20, f"Débit séquentiel trop bas : {rps:.1f} req/s"