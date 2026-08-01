"""
Script d'export du modèle final vers des artefacts autonomes (indépendants de MLflow),
utilisables par l'API sans dépendre d'un serveur MLflow accessible au runtime.

Script AUTONOME — s'exécute directement, sans dépendre de variables de notebook :

    python scripts/export_model.py

Prérequis :
    - Le modèle a été entraîné et enregistré dans MLflow sous le nom
      "scoring_credit_lgbm_optimise" (étape 4).
    - Le fichier output/seuil_optimal.txt existe (généré par l'étape 4).
"""

import json
import os

import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

# ── Configuration ─────────────────────────────────────────────────────
OUTPUT_DIR = "./output"          # adapter selon l'emplacement d'exécution
MODEL_DIR = "./model"
MODEL_NAME = "scoring_credit_lgbm_optimise"

os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# ── 1. Récupération de la dernière version du modèle enregistré ────────
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    raise RuntimeError(
        f"Aucune version trouvée pour le modèle '{MODEL_NAME}'. "
        "Avez-vous bien exécuté le notebook étape 4 jusqu'au bout ?"
    )
latest_version = max(versions, key=lambda v: int(v.version))
model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"

lgbm_model = mlflow.lightgbm.load_model(model_uri)
booster = lgbm_model.booster_

# ── 2. Liste des features (stockée nativement dans le booster) ─────────
features = booster.feature_name()

# ── 3. Seuil de décision optimal (sauvegardé à l'étape 4) ──────────────
seuil_path = os.path.join(OUTPUT_DIR, "seuil_optimal.txt")
if not os.path.exists(seuil_path):
    raise FileNotFoundError(
        f"{seuil_path} introuvable. Il doit être généré par le notebook étape 4."
    )
with open(seuil_path) as f:
    seuil_optimal = float(f.read())

# ── 4. Métriques du run associé (pour metadata.json) ────────────────────
run = client.get_run(latest_version.run_id)
mean_auc = run.data.metrics.get("mean_auc", -1.0)
methode_optimisation = run.data.params.get("methode_optimisation", "unknown")

# ── Export des artefacts ────────────────────────────────────────────────
booster.save_model(os.path.join(MODEL_DIR, "lgbm_model.txt"))

with open(os.path.join(MODEL_DIR, "features.json"), "w") as f:
    json.dump(features, f, indent=2)

with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
    json.dump({"seuil_optimal": round(seuil_optimal, 4)}, f, indent=2)

metadata = {
    "model_name": MODEL_NAME,
    "model_version": latest_version.version,
    "run_id": latest_version.run_id,
    "n_features": len(features),
    "seuil_optimal": round(seuil_optimal, 4),
    "mean_auc": round(mean_auc, 4),
    "methode_optimisation": methode_optimisation,
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Export terminé :")
print(f"   - {MODEL_DIR}/lgbm_model.txt  (version {latest_version.version}, run {latest_version.run_id})")
print(f"   - {MODEL_DIR}/features.json  ({len(features)} features)")
print(f"   - {MODEL_DIR}/threshold.json (seuil = {seuil_optimal:.4f})")
print(f"   - {MODEL_DIR}/metadata.json")