# locustfile.py
"""
Test de charge avec Locust.
Lancer l'API dans un terminal : uvicorn app.main:app --port 8000
Puis : locust -f locustfile.py --host=http://127.0.0.1:8000
Ouvrir http://localhost:8089 pour piloter.
"""
from locust import HttpUser, between, task


PAYLOAD = {
    "age": 35,
    "revenu_mensuel": 250_000,
    "montant_credit": 1_000_000,
    "duree_mois": 12,
    "anciennete_activite": 4,
}


class PredictUser(HttpUser):
    wait_time = between(0.5, 1.0)   # pause 0.5-1 s entre requêtes

    @task(10)
    def predict(self):
        self.client.post("/predict", json=PAYLOAD, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")