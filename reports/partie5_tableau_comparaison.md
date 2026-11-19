# Rapport de non-régression — v1.0.0 vs v1.0.0 (courante)

## Configuration
- Date : 2025-01-XX
- Commit de référence : <hash>
- Modèle : <version>

## Métriques

| Métrique | Baseline | Courante | Écart | Tolérance | Statut |
|---|---|---|---|---|---|
| Rappel | 0.600 | 0.600 | 0.000 | -0.020 | ✅ OK |
| Précision | 0.750 | 0.750 | 0.000 | -0.030 | ✅ OK |
| F1 | 0.667 | 0.667 | 0.000 | -0.020 | ✅ OK |

## Sorties de prédiction (20 profils)
- Écart max : 0.0e+00 (≤ 1e-6) ✅
- Toutes dans [0,1] ✅

## Cohérence save/load
- Écart max : 0.0e+00 (≤ 1e-9) ✅

## Démonstration de détection
- Dégradation simulée (`proba * 1.5`) → **3 tests FAILED** ✅ (chaîne opérationnelle)
- Après retour arrière → **10 tests PASSED** ✅