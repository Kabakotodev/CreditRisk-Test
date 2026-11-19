# app/preprocessing.py
import numpy as np


FEATURES = ["age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite"]


def transform(payload: dict) -> np.ndarray:
    """Transforme un payload validé en matrice (1, n_features)."""
    row = [float(payload[f]) for f in FEATURES]
    return np.array([row], dtype=float)