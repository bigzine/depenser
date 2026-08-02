"""
API de scoring crédit — "Prêt à Dépenser"

Lancement local :
    uvicorn api.main:app --reload --port 8000

Documentation interactive une fois lancée :
    http://127.0.0.1:8000/docs
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse

from .model_loader import scoring_model
from .schemas import (
    ClientData,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scoring_api")

# ── Logging structuré des requêtes de prédiction (pour analyse de drift) ────
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
PREDICTIONS_LOG_PATH = os.path.join(LOG_DIR, "predictions.jsonl")

# Clé simple pour protéger l'endpoint d'export des logs (PoC — à remplacer par
# une vraie gestion de secrets/authentification en production).
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "changeme")


def log_prediction_event(
    sk_id_curr,
    features: dict,
    proba: float | None,
    decision: str | None,
    threshold: float | None,
    missing_features: int | None,
    inference_time_ms: float,
    status: str,
    error_detail: str | None = None,
) -> None:
    """Écrit un évènement de prédiction au format JSON Lines pour analyse ultérieure."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sk_id_curr": sk_id_curr,
        "status": status,  # "success" ou "error"
        "inputs": features,
        "n_inputs_provided": len(features) if features else 0,
        "probability_default": proba,
        "decision": decision,
        "threshold_used": threshold,
        "missing_features": missing_features,
        "inference_time_ms": inference_time_ms,
        "error_detail": error_detail,
    }
    try:
        with open(PREDICTIONS_LOG_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:  # noqa: BLE001
        logger.exception("Échec de l'écriture du log de prédiction")


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

    Chaque appel (succès ou erreur) est journalisé dans logs/predictions.jsonl
    pour permettre une analyse ultérieure du data drift et des métriques
    opérationnelles (latence, taux d'erreur).
    """
    features = client.features.to_feature_dict()

    if not scoring_model.is_loaded:
        log_prediction_event(
            client.sk_id_curr, features, None, None, None, None, 0.0,
            status="error", error_detail="Modèle non chargé",
        )
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    start = time.perf_counter()
    try:
        proba = scoring_model.predict_proba(features)
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception("Erreur lors de la prédiction")
        log_prediction_event(
            client.sk_id_curr, features, None, None, scoring_model.threshold, None,
            elapsed_ms, status="error", error_detail=str(exc),
        )
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

    log_prediction_event(
        client.sk_id_curr, features, response.probability_default, response.decision,
        response.threshold_used, missing, elapsed_ms, status="success",
    )

    return response


@app.get("/admin/logs", response_class=PlainTextResponse, tags=["Admin"])
def export_logs(x_admin_key: str = Header(default="")):
    """
    Exporte les logs de prédiction accumulés (format JSON Lines), pour
    téléchargement et analyse locale (drift, anomalies opérationnelles).

    PoC : protection minimale par clé statique dans un header. À remplacer
    par une authentification robuste avant tout usage en production réelle.
    """
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Clé d'administration invalide.")

    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return ""

    with open(PREDICTIONS_LOG_PATH) as f:
        return f.read()


@app.get("/admin/logs/stats", tags=["Admin"])
def logs_stats(x_admin_key: str = Header(default="")):
    """Statistiques rapides sur les logs accumulés (sans tout télécharger)."""
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Clé d'administration invalide.")

    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return {"n_events": 0}

    n_events = 0
    n_errors = 0
    with open(PREDICTIONS_LOG_PATH) as f:
        for line in f:
            n_events += 1
            if '"status": "error"' in line:
                n_errors += 1

    return {"n_events": n_events, "n_errors": n_errors, "log_path": PREDICTIONS_LOG_PATH}