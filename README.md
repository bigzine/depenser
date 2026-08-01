# Prêt à Dépenser — Scoring Crédit

Projet de modélisation et mise en production d'un modèle de scoring crédit
pour l'entreprise "Prêt à Dépenser".

## Structure du projet

```
├── notebooks/          # Notebooks d'analyse et d'entraînement (étapes 1 à 4)
├── api/                 # API FastAPI de scoring (en cours de développement)
├── scripts/              # Scripts utilitaires (export du modèle, etc.)
├── tests/                # Tests unitaires (à venir)
├── model/                # Artefacts du modèle exporté (généré par scripts/export_model.py)
├── requirements.txt       # Dépendances Python
└── .gitignore
```

## Étapes du projet

1. **Préparation des données** (`notebooks/etape1_preparation_donnees.ipynb`)
2. **Tracking MLflow** (`notebooks/etape2_mlflow_tracking.ipynb`)
3. **Comparaison des modèles** (`notebooks/etape3_comparaison_modeles.ipynb`)
4. **Optimisation hyperparamètres & seuil métier** (`notebooks/etape4_optimisation_hp_seuil.ipynb`)
5. **API de scoring** (`api/`) — en cours
6. **Conteneurisation Docker** — à venir
7. **Monitoring & data drift** — à venir
8. **CI/CD** — à venir

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Modèle

Le modèle final est un **LightGBM** optimisé via Optuna, sélectionné pour son
meilleur compromis AUC / coût métier parmi 5 familles de modèles testées
(régression logistique, random forest, LightGBM, XGBoost, MLP).

- AUC (validation croisée) : ~0.778
- Seuil de décision optimal : ~0.502
- Coût métier : `10 × FN + 1 × FP` (un défaut manqué coûte 10× plus cher
  qu'un bon client refusé à tort)

## Documentation complète

Une documentation détaillée (lancement de l'API, interprétation du monitoring)
sera ajoutée une fois ces composants finalisés.
