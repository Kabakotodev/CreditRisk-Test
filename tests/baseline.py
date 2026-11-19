# tests/baseline.py
"""
Baseline de référence pour les tests de non-régression.
Versionnée dans le dépôt (baseline_v1.json).
"""
from pathlib import Path
import json

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


BASELINE_PATH = Path(__file__).parent.parent / "baseline_v1.json"


# Jeu de référence — 20 profils couvrant tout le spectre
REFERENCE_PROFILES = [
    # (age, revenu, montant, duree, anciennete)
    (25, 100_000,   300_000,   6,  1),
    (28, 150_000,   500_000,  12,  2),
    (30, 200_000,   500_000,  12,  3),
    (32, 220_000,   800_000,  18,  4),
    (35, 250_000, 1_000_000,  12,  4),
    (38, 300_000, 1_200_000,  24,  6),
    (40, 350_000, 1_500_000,  24,  8),
    (42, 400_000, 1_800_000,  36,  9),
    (45, 500_000, 1_500_000,  24,  8),
    (48, 550_000, 2_000_000,  36, 12),
    (50, 600_000, 2_500_000,  36, 15),
    (55, 800_000, 2_000_000,  36, 15),
    (60, 900_000, 3_000_000,  48, 20),
    (22,  80_000,   200_000,   6,  0),
    (70, 400_000,   500_000,  12, 30),
    (33, 180_000, 4_500_000,  60,  5),   # montant très élevé
    (27, 120_000,   100_000,   3,  1),   # petit crédit court
    (65, 700_000, 2_000_000,  24, 25),
    (36, 280_000, 1_000_000,  18,  5),
    (44, 450_000, 1_700_000,  30, 10),
]

# Labels "vrais" (simulés pour le calcul de rappel/F1)
# 1 = défaut attendu, 0 = remboursement attendu
REFERENCE_LABELS = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 1, 0, 0, 0]


def profiles_to_payloads(profiles=REFERENCE_PROFILES):
    """Convertit les tuples en dicts prêts pour l'API."""
    keys = ("age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite")
    return [dict(zip(keys, p)) for p in profiles]


def save_baseline(predictions: list[float], metrics: dict) -> None:
    """Sauvegarde la baseline versionnée."""
    data = {
        "version": "1.0.0",
        "predictions": predictions,
        "metrics": metrics,
        "profiles": REFERENCE_PROFILES,
        "labels": REFERENCE_LABELS,
    }
    BASELINE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {}
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def compute_metrics(predictions: list[float], labels=REFERENCE_LABELS,
                    threshold: float = 0.5) -> dict:
    """Calcule les métriques binaires à partir des probabilités."""
    y_pred = [1 if p >= threshold else 0 for p in predictions]
    y_true = list(labels)
    return {
        "rappel": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }