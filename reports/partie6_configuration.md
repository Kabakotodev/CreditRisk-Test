# Configuration des tests de performance

## Matériel
- Machine : <modèle ou "VM locale">
- CPU : <ex. Intel i7-1165G7 @ 2.80 GHz (4 cœurs, 8 threads)>
- RAM : <ex. 16 Go>
- Disque : <ex. SSD NVMe>
- OS : Windows 11 (ou Linux)

## Logiciel
- Python : 3.11.0
- FastAPI : 0.115.0
- Uvicorn : 0.30.6 (mode `--workers 1`)
- scikit-learn : 1.5.2
- Locust : 2.29.1
- Modèle : régression logistique factice (5 features)

## Conditions
- Machine isolée (pas d'autres charges)
- API lancée en local (`127.0.0.1:8000`)
- Modèle pré-chargé en mémoire
- Aucun proxy, aucun réseau