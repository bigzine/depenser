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
import pandas as pd

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

        with open(threshold_path) as f:
            self.threshold = json.load(f)["seuil_optimal"]

        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)

    @property
    def is_loaded(self) -> bool:
        return self.booster is not None

    def _build_row(self, features: Dict[str, float]) -> pd.DataFrame:
        """
        Construit une ligne de features dans l'ordre exact attendu par le modèle.
        Les colonnes manquantes deviennent NaN (LightGBM les gère nativement).
        """
        row = {col: features.get(col, np.nan) for col in self.features}
        return pd.DataFrame([row], columns=self.features)

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