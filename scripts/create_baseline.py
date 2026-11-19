# scripts/create_baseline.py
"""
À exécuter UNE SEULE FOIS, en début de mission, pour créer la baseline.
Ensuite, la baseline est committée dans le dépôt.
"""
import sys
from pathlib import Path

# Permettre l'import de app/ et tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import prediction
from tests.baseline import (
    profiles_to_payloads, REFERENCE_LABELS, save_baseline, compute_metrics
)


def main():
    # 1. Charger un modèle (factice ou réel — ici, chargement réel si présent)
    if not prediction.load_model():
        print("Erreur : aucun modèle disponible. Générez model.joblib.")
        sys.exit(1)

    # 2. Prédire sur le jeu de référence
    from app.preprocessing import transform
    predictions = []
    for payload in profiles_to_payloads():
        X = transform(payload)
        predictions.append(prediction.predict_proba(X))

    # 3. Calculer les métriques
    metrics = compute_metrics(predictions, REFERENCE_LABELS)

    # 4. Sauvegarder
    save_baseline(predictions, metrics)

    print(f"Baseline créée : {len(predictions)} prédictions")
    print(f"Métriques : {metrics}")


if __name__ == "__main__":
    main()