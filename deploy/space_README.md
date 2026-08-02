---
title: Scoring Credit API
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# API de Scoring Crédit — Prêt à Dépenser

API FastAPI exposant un modèle LightGBM de scoring crédit.

Déployée automatiquement depuis le dépôt GitHub via GitHub Actions
à chaque push sur `main`.

## Endpoints

- `GET /health` — vérifie que l'API et le modèle sont chargés
- `GET /model-info` — métadonnées du modèle (AUC, seuil, version)
- `POST /predict` — prédit la probabilité de défaut d'un client
- `GET /docs` — documentation interactive Swagger

## Documentation complète

Voir le dépôt GitHub source pour la documentation complète du projet
(notebooks, tests, pipeline CI/CD).
