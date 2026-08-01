"""Schémas de requête/réponse de l'API de scoring crédit."""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class ClientData(BaseModel):
    """
    Données d'un client à scorer.

    `features` est un dictionnaire {nom_de_colonne: valeur}. Les colonnes
    manquantes seront traitées comme valeurs manquantes (NaN), que LightGBM
    gère nativement. Les colonnes en trop (non utilisées par le modèle)
    sont simplement ignorées.
    """

    sk_id_curr: Optional[int] = Field(
        default=None, description="Identifiant client (optionnel, pour traçabilité)"
    )
    features: Dict[str, float] = Field(
        ..., description="Dictionnaire des features du client"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sk_id_curr": 100001,
                "features": {
                    "EXT_SOURCE_MEAN": 0.51,
                    "EXT_SOURCE_2": 0.52,
                    "EXT_SOURCE_3": 0.48,
                    "AMT_CREDIT": 406597.5,
                    "AMT_ANNUITY": 24700.5,
                },
            }
        }


class PredictionResponse(BaseModel):
    sk_id_curr: Optional[int] = None
    probability_default: float = Field(
        ..., description="Probabilité prédite de défaut (classe 1)"
    )
    decision: str = Field(
        ..., description="'ACCEPTE' si probabilité < seuil, sinon 'REFUSE'"
    )
    threshold_used: float
    inference_time_ms: float
    missing_features: int = Field(
        ..., description="Nombre de features attendues absentes de la requête"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    n_features: int
    seuil_optimal: float
    mean_auc: float
    methode_optimisation: str
