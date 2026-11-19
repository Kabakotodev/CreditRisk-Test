# Installation
pip install -r requirements.txt

# Exécution ciblée Partie II
pytest tests/test_api.py -v

# Avec rapport lisible pour capture d'écran
pytest tests/test_api.py -v --tb=short > reports/partie2_api.txt

