# Rapport de performance — v1.0.0

## Configuration
(cf. `reports/partie6_configuration.md`)

## Objectifs
| Métrique | Cible |
|---|---|
| Latence P95 | < 200 ms |
| Latence moyenne | < 100 ms |
| Débit soutenu | ≥ 50 req/s |
| Latence inférence | < 10 ms |
| Taux d'erreur | < 1 % |

## Mesures (pytest)
| Métrique | Valeur | Statut |
|---|---|---|
| Latence min | ... ms | — |
| Latence moyenne | ... ms | ✅/❌ |
| Latence médiane | ... ms | — |
| **Latence P95** | ... ms | ✅/❌ |
| Latence max | ... ms | — |
| Latence inférence P95 | ... ms | ✅/❌ |
| Débit séquentiel | ... req/s | — |

## Mesures (Locust — charge progressive)
| Utilisateurs | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Taux d'erreur |
|---|---|---|---|---|---|
| 10 | ... | ... | ... | ... | ... |
| 50 | ... | ... | ... | ... | ... |
| 100 | ... | ... | ... | ... | ... |

## Interprétation
- ✅/❌ Les objectifs sont atteints : ...
- Comportement sous charge : ...
- Points d'attention : ...
- Recommandations : ...