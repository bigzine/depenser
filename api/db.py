"""
Couche de persistance des logs de prédiction.

Utilise SQLAlchemy Core (pas l'ORM complet, inutile ici) pour que le même
code fonctionne indifféremment avec :
  - PostgreSQL en production (variable d'environnement DATABASE_URL, ex.
    fournie par Neon/Supabase/Railway)
  - SQLite en local/tests si DATABASE_URL n'est pas définie (fichier
    logs/predictions.db) ou en mémoire pour les tests (sqlite:///:memory:)

Écriture en base à chaque appel /predict : la donnée est persistée
immédiatement, sans étape d'export manuelle ou planifiée.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    select,
)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///" + os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.db"),
)

# SQLite nécessite ce flag pour être utilisé depuis plusieurs threads (FastAPI/Starlette
# peut traiter des requêtes sur des threads différents). Sans effet sur PostgreSQL.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

if DATABASE_URL.startswith("sqlite:///") and DATABASE_URL != "sqlite:///:memory:":
    db_path = DATABASE_URL.replace("sqlite:///", "")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("sk_id_curr", Integer, nullable=True),
    Column("status", String(20), nullable=False),
    Column("inputs", Text, nullable=False),  # JSON sérialisé
    Column("n_inputs_provided", Integer, nullable=False),
    Column("probability_default", Float, nullable=True),
    Column("decision", String(20), nullable=True),
    Column("threshold_used", Float, nullable=True),
    Column("missing_features", Integer, nullable=True),
    Column("inference_time_ms", Float, nullable=False),
    Column("error_detail", Text, nullable=True),
)


def init_db() -> None:
    """Crée la table si elle n'existe pas encore. Appelé au démarrage de l'API."""
    metadata.create_all(engine)


def log_prediction_event(
    sk_id_curr: Optional[int],
    features: dict,
    proba: Optional[float],
    decision: Optional[str],
    threshold: Optional[float],
    missing_features: Optional[int],
    inference_time_ms: float,
    status: str,
    error_detail: Optional[str] = None,
) -> None:
    """Insère un évènement de prédiction en base, immédiatement persisté."""
    with engine.begin() as conn:
        conn.execute(
            predictions.insert().values(
                timestamp=datetime.now(timezone.utc),
                sk_id_curr=sk_id_curr,
                status=status,
                inputs=json.dumps(features),
                n_inputs_provided=len(features) if features else 0,
                probability_default=proba,
                decision=decision,
                threshold_used=threshold,
                missing_features=missing_features,
                inference_time_ms=inference_time_ms,
                error_detail=error_detail,
            )
        )


def export_logs_as_jsonl() -> str:
    """Exporte tous les évènements au format JSON Lines (pour compatibilité
    avec les notebooks/dashboard existants, qui consomment ce format)."""
    with engine.connect() as conn:
        rows = conn.execute(select(predictions).order_by(predictions.c.timestamp)).mappings().all()

    lines = []
    for row in rows:
        event = dict(row)
        event["timestamp"] = event["timestamp"].isoformat()
        event["inputs"] = json.loads(event["inputs"])
        event.pop("id", None)
        lines.append(json.dumps(event))
    return "\n".join(lines) + ("\n" if lines else "")


def get_stats() -> dict:
    """Statistiques rapides sans charger l'intégralité des logs en mémoire."""
    with engine.connect() as conn:
        n_total = conn.execute(select(func.count()).select_from(predictions)).scalar()
        n_errors = conn.execute(
            select(func.count()).select_from(predictions).where(predictions.c.status == "error")
        ).scalar()
    return {"n_events": n_total or 0, "n_errors": n_errors or 0}
