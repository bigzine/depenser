"""
API de scoring crédit — "Prêt à Dépenser"

Lancement local :
    uvicorn api.main:app --reload --port 8000

Documentation interactive une fois lancée :
    http://127.0.0.1:8000/docs
"""

import logging
import time

from fastapi import FastAPI, HTTPException

from .model_loader import scoring_model
from .schemas import (
    ClientData,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scoring_api")

app = FastAPI(
    title="Prêt à Dépenser — API de Scoring Crédit",
    description="Prédit la probabilité de défaut d'un client et la décision d'octroi de crédit.",
    version="1.0.0",
)


@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "API de scoring crédit — Prêt à Dépenser",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Vérifie que l'API est démarrée et que le modèle est bien chargé."""
    return HealthResponse(status="ok", model_loaded=scoring_model.is_loaded)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Meta"])
def model_info():
    """Retourne les métadonnées du modèle actuellement chargé."""
    meta = scoring_model.metadata
    if not meta:
        raise HTTPException(status_code=404, detail="Métadonnées du modèle indisponibles.")
    return ModelInfoResponse(
        model_name=meta.get("model_name", "unknown"),
        n_features=meta.get("n_features", len(scoring_model.features)),
        seuil_optimal=meta.get("seuil_optimal", scoring_model.threshold),
        mean_auc=meta.get("mean_auc", -1.0),
        methode_optimisation=meta.get("methode_optimisation", "unknown"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict(client: ClientData):
    """
    Prédit la probabilité de défaut d'un client et retourne la décision
    d'octroi de crédit (ACCEPTE / REFUSE) selon le seuil métier optimal.

    Validation :
      - EXT_SOURCE_MEAN et AMT_CREDIT sont obligatoires (422 si absents).
      - Les champs bornés (EXT_SOURCE_*, AGE_YEARS, REGION_RATING_CLIENT, montants...)
        sont validés automatiquement par le schéma (422 si hors plage ou mauvais type).
      - Les autres features (parmi les 200 attendues) sont optionnelles.
    """
    if not scoring_model.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    features = client.features.to_feature_dict()

    start = time.perf_counter()
    try:
        proba = scoring_model.predict_proba(features)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erreur lors de la prédiction")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {exc}") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000

    missing = scoring_model.count_missing(features)
    if missing:
        logger.warning(
            "Client %s : %d/%d features attendues sont absentes de la requête",
            client.sk_id_curr, missing, len(scoring_model.features),
        )

    response = PredictionResponse(
        sk_id_curr=client.sk_id_curr,
        probability_default=round(proba, 4),
        decision=scoring_model.decide(proba),
        threshold_used=scoring_model.threshold,
        inference_time_ms=round(elapsed_ms, 2),
        missing_features=missing,
    )

    logger.info(
        "sk_id_curr=%s proba=%.4f decision=%s missing=%d time_ms=%.2f",
        client.sk_id_curr, proba, response.decision, missing, elapsed_ms,
    )

    return response