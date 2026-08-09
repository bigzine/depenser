# Prêt à Dépenser — Scoring Crédit

Modèle de scoring crédit pour l'entreprise "Prêt à Dépenser", de l'entraînement
à la mise en production : API, conteneurisation Docker, pipeline CI/CD, et
monitoring du data drift en production.

**API en production** : https://ediagabate-scoring-credit-api.hf.space
**Documentation interactive (Swagger)** : https://ediagabate-scoring-credit-api.hf.space/docs

---

## Structure du projet

```
├── notebooks/                     # Entraînement et analyse (étapes 1 à 4)
│   ├── etape1_preparation_donnees.ipynb
│   ├── etape2_mlflow_tracking.ipynb
│   ├── etape3_comparaison_modeles.ipynb
│   └── etape4_optimisation_hp_seuil.ipynb
├── api/                           # API FastAPI de scoring
│   ├── main.py                    # Endpoints (health, model-info, predict, admin/logs)
│   ├── model_loader.py            # Chargement du modèle (une seule fois au démarrage)
│   └── schemas.py                 # Validation Pydantic (champs obligatoires, plages, types)
├── scripts/
│   └── export_model.py            # Exporte le modèle MLflow → artefacts autonomes (model/)
├── model/                         # Artefacts du modèle exporté (utilisés par l'API)
│   ├── lgbm_model.txt
│   ├── features.json
│   ├── threshold.json
│   └── metadata.json
├── tests/                         # Suite pytest (unitaires + intégration)
├── monitoring/                    # Analyse du data drift et dashboard
│   ├── simulate_traffic.py        # Génère du trafic de test vers l'API déployée
│   ├── etape3_drift_analysis.ipynb  # Analyse de drift (étude complète, avec résultats)
│   ├── dashboard.py                # Dashboard Streamlit interactif
│   └── production_logs.jsonl       # Exemple de logs de production exportés
├── .github/workflows/ci-cd.yml    # Pipeline CI/CD (test → build → déploiement)
├── deploy/space_README.md         # Métadonnées du Space Hugging Face
├── Dockerfile
├── requirements.txt               # Dépendances de l'API (image Docker)
├── requirements-dev.txt           # + dépendances notebooks/tests/monitoring
└── .gitignore
```

---

## 1. Installation locale

```bash
git clone <url-du-depot>
cd pret-a-depenser-scoring

python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate sous Windows

pip install -r requirements-dev.txt   # inclut requirements.txt (API) + outils de dev
```

## 2. Le modèle

Modèle final : **LightGBM** optimisé via Optuna (étape 4), sélectionné pour son
meilleur compromis AUC / coût métier parmi 5 familles de modèles comparées
(régression logistique, random forest, LightGBM, XGBoost, MLP).

- AUC (validation croisée) : ~0.778
- Seuil de décision optimal : ~0.502
- Fonction de coût métier : `10 × FN + 1 × FP` (un défaut manqué coûte 10× plus
  cher qu'un bon client refusé à tort)
- 200 features sélectionnées par corrélation, avec `class_weight="balanced"`
  pour gérer le déséquilibre des classes

Le modèle est suivi et versionné avec **MLflow** (base locale `mlflow.db`,
non versionnée dans git — voir `.gitignore`).

### Exporter le modèle pour l'API

L'API ne dépend pas de MLflow au runtime (pour rester légère et déployable en
conteneur). Après avoir entraîné/enregistré le modèle final (notebook étape 4),
générez les artefacts autonomes :

```bash
python scripts/export_model.py
```

Cela produit `model/lgbm_model.txt`, `model/features.json`,
`model/threshold.json` et `model/metadata.json`, utilisés directement par l'API.

## 3. Lancer l'API

### En local (sans Docker)

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Documentation interactive : http://127.0.0.1:8000/docs

### Avec Docker

```bash
docker build -t scoring-api .
docker run -p 8000:8000 scoring-api
```

Variable d'environnement utile :
- `ADMIN_API_KEY` — clé protégeant les endpoints `/admin/logs*` (par défaut
  `changeme`, **à définir explicitement en production**) :
  ```bash
  docker run -p 8000:8000 -e ADMIN_API_KEY="votre_clé_secrète" scoring-api
  ```

### Endpoints principaux

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | Vérifie que l'API et le modèle sont chargés |
| `/model-info` | GET | Métadonnées du modèle (AUC, seuil, méthode d'optimisation) |
| `/predict` | POST | Prédit la probabilité de défaut d'un client |
| `/admin/logs` | GET | Exporte les logs de prédiction (protégé par `x-admin-key`) |
| `/admin/logs/stats` | GET | Statistiques rapides sur les logs (protégé) |
| `/docs` | GET | Documentation interactive Swagger |

### Exemple de requête `/predict`

```bash
curl -X POST https://ediagabate-scoring-credit-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sk_id_curr": 100001,
    "features": {
      "EXT_SOURCE_MEAN": 0.51,
      "AMT_CREDIT": 406597.5,
      "EXT_SOURCE_2": 0.52,
      "EXT_SOURCE_3": 0.48,
      "AGE_YEARS": 35,
      "AMT_ANNUITY": 24700.5,
      "additional_features": {
        "REGION_POPULATION_RELATIVE": 0.018
      }
    }
  }'
```

**Champs obligatoires** : `EXT_SOURCE_MEAN`, `AMT_CREDIT`.
**Champs avec validation de plage** : scores `EXT_SOURCE_*` (0–1), `AGE_YEARS`
(18–100), `REGION_RATING_CLIENT` (1–3), montants (`AMT_CREDIT`, `AMT_ANNUITY`,
`AMT_GOODS_PRICE` > 0). Toute autre feature parmi les 200 attendues peut être
passée dans `additional_features`.

## 4. Tests

```bash
python -m pytest tests/ -v
```

36 tests unitaires + intégration : validation du schéma (champs obligatoires,
plages, types), endpoints (`/health`, `/model-info`, `/predict`), et
journalisation des prédictions.

## 5. CI/CD

Le pipeline (`.github/workflows/ci-cd.yml`) s'exécute à chaque push sur `main` :

1. **test** — installe les dépendances et exécute la suite pytest
2. **build** — construit l'image Docker et vérifie que le conteneur démarre
   correctement (`/health` répond)
3. **deploy** — (uniquement sur `main`) pousse le code vers le Space Hugging
   Face, qui reconstruit et redéploie automatiquement l'API

Secret requis dans GitHub (Settings → Secrets and variables → Actions) :
`HF_TOKEN` (token Hugging Face avec droits d'écriture).

## 6. Monitoring et data drift

### Comment sont collectées les données de production

Chaque appel à `/predict` (succès ou erreur) est journalisé au format JSON
Lines dans `logs/predictions.jsonl` côté API : timestamp, features en entrée,
prédiction, décision, latence, nombre de features manquantes.

Le stockage de l'API Hugging Face étant éphémère (pas de persistance entre
redémarrages), le flux recommandé est :

```bash
# 1. Générer du trafic (ou attendre du trafic réel)
python monitoring/simulate_traffic.py --url https://ediagabate-scoring-credit-api.hf.space \
    --n-normal 150 --n-drifted 150

# 2. Exporter les logs accumulés
curl https://ediagabate-scoring-credit-api.hf.space/admin/logs \
  -H "x-admin-key: <votre_clé>" > monitoring/production_logs.jsonl
```

### Analyser le drift

**Notebook complet** (étude détaillée, déjà exécuté avec résultats) :
```bash
jupyter notebook monitoring/etape3_drift_analysis.ipynb
```

**Dashboard interactif** :
```bash
streamlit run monitoring/dashboard.py
```
Le dashboard accepte soit un fichier `.jsonl` local, soit une récupération en
direct depuis l'API (URL + clé admin).

### Interpréter les résultats

- **Test statistique** : Kolmogorov-Smirnov (K-S) pour les features
  numériques, test du chi² d'ajustement pour les features catégorielles.
  Seuil de significativité : p-value < 0.05 → drift détecté sur cette feature.
- **% de features en drift** : une proportion élevée (ex. > 50%) sur les
  features les plus prédictives (`EXT_SOURCE_*`) est un signal fort justifiant
  une investigation, voire un ré-entraînement du modèle.
- **Point de vigilance — tests multiples** : avec plusieurs features testées
  simultanément à un seuil de 0.05, un faux positif isolé est statistiquement
  attendu ~1 fois sur 20. Ne pas déclencher d'alerte automatique sur une seule
  feature proche du seuil sans corroboration.
- **Data drift ≠ concept drift** : cette analyse détecte un changement dans la
  distribution des features en entrée, pas directement une baisse de
  performance du modèle. Confirmer une dégradation réelle nécessite de
  comparer les prédictions aux vrais résultats de défaut une fois connus.
- **Métriques opérationnelles** : taux d'erreur (alerter si > quelques %) et
  latence p95/p99 (alerter si anormalement élevée par rapport à la baseline
  observée, de l'ordre de quelques ms à quelques dizaines de ms en usage normal).

### Limites connues (PoC)

- La référence utilisée pour le drift est une fenêtre de trafic simulé (basée
  sur les statistiques réelles d'entraînement), pas directement `X_train`.
  En production, comparer directement à `X_train` est recommandé.
- Le dashboard utilise `scipy.stats` directement plutôt que la librairie
  `evidently`, en raison d'un bug de compatibilité connu et non résolu
  d'`evidently` avec Python 3.13
  ([evidentlyai/evidently#1517](https://github.com/evidentlyai/evidently/issues/1517)).
  Les valeurs de p-value ont été vérifiées identiques à celles produites par
  `evidently` sur les mêmes données.
- Les logs contiennent des données personnelles (âge, montants). En
  production réelle, prévoir anonymisation/pseudonymisation et une politique
  de rétention conforme RGPD.

## 7. Historique et gouvernance

- Commits explicites par fonctionnalité (voir `git log`)
- `.gitignore` excluant données, base MLflow locale, secrets et environnements
  virtuels
- Secrets gérés via GitHub Actions Secrets (`HF_TOKEN`) et variables
  d'environnement Hugging Face (`ADMIN_API_KEY`), jamais commités dans le code