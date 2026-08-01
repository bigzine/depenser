"""
Script d'export du modèle final vers des artefacts autonomes (indépendants de MLflow),
utilisables par l'API sans dépendre d'un serveur MLflow accessible au runtime.

À exécuter UNE FOIS depuis le notebook étape 4 (etape4_optimisation_hp_seuil.ipynb),
après la cellule qui entraîne `lgbm_final` et calcule `seuil_opt` / `TOP_200`.

Copiez ce code dans une nouvelle cellule à la fin du notebook, ou exécutez-le
en important les variables nécessaires si vous préférez un script séparé.
"""

import json
import os

# ── Dossier de sortie ────────────────────────────────────────────────
MODEL_DIR = "../model"  # notebook dans notebooks/, model/ à la racine du repo
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Modèle LightGBM (format natif, pas de dépendance pickle/mlflow) ─
lgbm_final.booster_.save_model(os.path.join(MODEL_DIR, "lgbm_model.txt"))

# ── 2. Liste des features attendues, dans l'ordre exact d'entraînement ─
with open(os.path.join(MODEL_DIR, "features.json"), "w") as f:
    json.dump(TOP_200, f, indent=2)

# ── 3. Seuil de décision optimal ────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
    json.dump({"seuil_optimal": round(seuil_opt, 4)}, f, indent=2)

# ── 4. Métadonnées (utile pour /model-info et le suivi de version) ─────
metadata = {
    "model_name": "scoring_credit_lgbm_optimise",
    "n_features": len(TOP_200),
    "seuil_optimal": round(seuil_opt, 4),
    "mean_auc": round(float(best_res_final["AUC"]), 4),
    "methode_optimisation": methode_gagnante,
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Export terminé :")
print(f"   - {MODEL_DIR}/lgbm_model.txt")
print(f"   - {MODEL_DIR}/features.json  ({len(TOP_200)} features)")
print(f"   - {MODEL_DIR}/threshold.json (seuil = {seuil_opt:.4f})")
print(f"   - {MODEL_DIR}/metadata.json")
