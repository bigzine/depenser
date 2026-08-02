"""
Fixtures partagées pour les tests.

IMPORTANT : le modèle factice est construit et MODEL_DIR est positionné au
niveau du module (donc exécuté une seule fois, avant la collecte des tests),
car api/model_loader.py charge le modèle une seule fois au moment de
l'import de api.main (voir docstring de ce module). Si on important
api.main avant d'avoir positionné MODEL_DIR, le modèle chargé serait le
mauvais (ou une erreur si model/ n'existe pas encore localement).
"""

import json
import os
import tempfile

import lightgbm as lgb
import numpy as np
import pytest

# ── Modèle factice avec de VRAIS noms de features (issus de model/features.json) ─
TEST_FEATURES = [
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_CREDIT",
    "AGE_YEARS",
    "REGION_RATING_CLIENT",
]

_tmp_dir = tempfile.mkdtemp(prefix="scoring_test_model_")

rng = np.random.default_rng(42)
X = rng.random((300, len(TEST_FEATURES)))
y = (X[:, 0] + X[:, 1] > 1).astype(int)

_booster = lgb.train(
    {"objective": "binary", "verbose": -1},
    lgb.Dataset(X, label=y, feature_name=TEST_FEATURES),
    num_boost_round=5,
)
_booster.save_model(os.path.join(_tmp_dir, "lgbm_model.txt"))

with open(os.path.join(_tmp_dir, "features.json"), "w") as f:
    json.dump(TEST_FEATURES, f)

with open(os.path.join(_tmp_dir, "threshold.json"), "w") as f:
    json.dump({"seuil_optimal": 0.5}, f)

with open(os.path.join(_tmp_dir, "metadata.json"), "w") as f:
    json.dump(
        {
            "model_name": "test_model",
            "n_features": len(TEST_FEATURES),
            "seuil_optimal": 0.5,
            "mean_auc": 0.77,
            "methode_optimisation": "test",
        },
        f,
    )

# Positionné AVANT tout import de api.main (voir api/model_loader.py : MODEL_DIR
# est lu via une variable d'environnement au moment de l'import du module).
os.environ["MODEL_DIR"] = _tmp_dir


@pytest.fixture(scope="session")
def client():
    """Client de test FastAPI, réutilisé pour tous les tests d'intégration."""
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


@pytest.fixture
def valid_payload():
    """Un payload client valide, satisfaisant tous les champs obligatoires et bornés."""
    return {
        "sk_id_curr": 100001,
        "features": {
            "EXT_SOURCE_MEAN": 0.51,
            "AMT_CREDIT": 406597.5,
            "EXT_SOURCE_2": 0.52,
            "EXT_SOURCE_3": 0.48,
            "AGE_YEARS": 35,
            "REGION_RATING_CLIENT": 2,
        },
    }
