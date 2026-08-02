"""Schémas de requête/réponse de l'API de scoring crédit."""

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClientFeatures(BaseModel):
    """
    Features d'un client. Un sous-ensemble de champs métier importants sont
    explicitement validés (obligatoires et/ou bornés) ; les ~185 features
    restantes (parmi les 200 utilisées par le modèle) passent par
    `additional_features`, sans validation métier spécifique.
    """

    # ── Champs obligatoires ──────────────────────────────────────────
    EXT_SOURCE_MEAN: float = Field(
        ..., ge=0, le=1,
        description="Moyenne des scores externes (obligatoire, entre 0 et 1)",
    )
    AMT_CREDIT: float = Field(
        ..., gt=0, description="Montant du crédit demandé (obligatoire, doit être positif)"
    )

    # ── Champs optionnels avec validation de plage ─────────────────────
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_MIN: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_PROD: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_STD: Optional[float] = Field(None, ge=0)
    AGE_YEARS: Optional[float] = Field(None, ge=18, le=100)
    EMPLOYED_YEARS: Optional[float] = Field(None, ge=0)
    REGION_RATING_CLIENT: Optional[int] = Field(None, ge=1, le=3)
    REGION_RATING_CLIENT_W_CITY: Optional[int] = Field(None, ge=1, le=3)
    AMT_ANNUITY: Optional[float] = Field(None, gt=0)
    AMT_GOODS_PRICE: Optional[float] = Field(None, gt=0)
    CC_UTILIZATION_RATE_MEAN: Optional[float] = Field(None, ge=0)

    # ── Toutes les autres features attendues par le modèle ─────────────
    additional_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Features restantes parmi les 200 attendues (nom -> valeur)",
    )

    def to_feature_dict(self) -> Dict[str, float]:
        """Aplati le modèle en un seul dict {nom_feature: valeur} pour le modèle."""
        known = self.model_dump(exclude={"additional_features"}, exclude_none=True)
        return {**known, **self.additional_features}


class ClientData(BaseModel):
    sk_id_curr: Optional[int] = Field(
        default=None, description="Identifiant client (optionnel, pour traçabilité)"
    )
    features: ClientFeatures

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sk_id_curr": 100001,
                "features": {
                    "EXT_SOURCE_MEAN": 0.51,
                    "AMT_CREDIT": 406597.5,
                    "EXT_SOURCE_2": 0.52,
                    "EXT_SOURCE_3": 0.48,
                    "AGE_YEARS": 35,
                    "AMT_ANNUITY": 24700.5,
                    "additional_features": {
                        "REGION_POPULATION_RELATIVE": 0.018,
                        "FLAG_DOCUMENT_3": 1,
                    },
                },
            }
        }
    )


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
        ..., description="Nombre de features attendues (parmi les 200) absentes de la requête"
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
