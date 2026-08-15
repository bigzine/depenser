"""
Chargement du modèle et de ses artefacts associés.

Les fichiers attendus dans MODEL_DIR (générés par scripts/export_model.py) :
  - lgbm_model.txt   : modèle LightGBM au format natif (booster)
  - features.json    : liste ordonnée des features attendues
  - threshold.json   : seuil de décision optimal
  - metadata.json     : métadonnées du modèle (AUC, méthode d'optimisation, etc.)
"""

import json
import os
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model"))


class ScoringModel:
    """Wrapper autour du modèle LightGBM chargé une seule fois au démarrage de l'API."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.booster: Optional[lgb.Booster] = None
        self.features: List[str] = []
        self.threshold: float = 0.5
        self.metadata: dict = {}
        self._load()

    def _load(self) -> None:
        model_path = os.path.join(self.model_dir, "lgbm_model.txt")
        features_path = os.path.join(self.model_dir, "features.json")
        threshold_path = os.path.join(self.model_dir, "threshold.json")
        metadata_path = os.path.join(self.model_dir, "metadata.json")

        for path in (model_path, features_path, threshold_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Artefact manquant : {path}. "
                    "Avez-vous exécuté scripts/export_model.py depuis le notebook étape 4 ?"
                )

        self.booster = lgb.Booster(model_file=model_path)

        with open(features_path) as f:
            self.features = json.load(f)
        # Index précalculé {nom_feature: position} pour un accès O(1) lors de
        # la construction de chaque ligne (voir _build_row).
        self._feature_index = {name: i for i, name in enumerate(self.features)}
        self._n_features = len(self.features)

        with open(threshold_path) as f:
            self.threshold = json.load(f)["seuil_optimal"]

        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)

    @property
    def is_loaded(self) -> bool:
        return self.booster is not None

    def _build_row(self, features: Dict[str, float]) -> np.ndarray:
        """
        Construit une ligne de features (tableau numpy 1×N) dans l'ordre exact
        attendu par le modèle. Les colonnes manquantes deviennent NaN
        (LightGBM les gère nativement).

        Optimisation (étape 4 — profiling) : construction directe en numpy,
        sans passer par un pandas.DataFrame intermédiaire. Le profiling a
        montré que la construction du DataFrame représentait ~73% du temps
        total de prédiction par requête (voir monitoring/etape4_optimisation_performance.ipynb),
        principalement à cause de la création d'objets pandas puis de leur
        reconversion interne en numpy par LightGBM lui-même. Cette version
        est ~28x plus rapide, avec des prédictions strictement identiques.
        """
        row = np.full(self._n_features, np.nan, dtype=np.float64)
        for key, value in features.items():
            idx = self._feature_index.get(key)
            if idx is not None:
                row[idx] = value
        return row.reshape(1, -1)

    def count_missing(self, features: Dict[str, float]) -> int:
        return sum(1 for col in self.features if col not in features)

    def predict_proba(self, features: Dict[str, float]) -> float:
        X = self._build_row(features)
        proba = self.booster.predict(X)[0]
        return float(proba)

    def decide(self, proba: float) -> str:
        return "REFUSE" if proba >= self.threshold else "ACCEPTE"


# Instance unique, chargée au démarrage du module (import) — un seul chargement
# pour toute la durée de vie du processus API.
scoring_model = ScoringModel()
