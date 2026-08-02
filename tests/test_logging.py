"""
Tests D'INTÉGRATION pour la journalisation structurée des prédictions et
les endpoints d'export (/admin/logs), utilisés pour l'analyse de drift.
"""

import json
import os


def test_predict_writes_to_log_file(client, valid_payload):
    from api.main import PREDICTIONS_LOG_PATH

    # État avant l'appel
    n_lines_before = 0
    if os.path.exists(PREDICTIONS_LOG_PATH):
        with open(PREDICTIONS_LOG_PATH) as f:
            n_lines_before = sum(1 for _ in f)

    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200

    with open(PREDICTIONS_LOG_PATH) as f:
        lines = f.readlines()
    assert len(lines) == n_lines_before + 1

    last_event = json.loads(lines[-1])
    assert last_event["status"] == "success"
    assert last_event["sk_id_curr"] == valid_payload["sk_id_curr"]
    assert last_event["decision"] in ("ACCEPTE", "REFUSE")
    assert last_event["probability_default"] is not None
    assert "inputs" in last_event
    assert last_event["inference_time_ms"] >= 0


def test_predict_validation_error_does_not_reach_logging(client):
    """Une erreur 422 (validation Pydantic) se produit avant l'appel au modèle :
    elle n'est pas journalisée comme évènement de prédiction (comportement attendu,
    car aucune tentative de scoring n'a eu lieu)."""
    from api.main import PREDICTIONS_LOG_PATH

    n_lines_before = 0
    if os.path.exists(PREDICTIONS_LOG_PATH):
        with open(PREDICTIONS_LOG_PATH) as f:
            n_lines_before = sum(1 for _ in f)

    r = client.post("/predict", json={"sk_id_curr": 1, "features": {}})
    assert r.status_code == 422

    n_lines_after = 0
    if os.path.exists(PREDICTIONS_LOG_PATH):
        with open(PREDICTIONS_LOG_PATH) as f:
            n_lines_after = sum(1 for _ in f)
    assert n_lines_after == n_lines_before


def test_admin_logs_requires_valid_key(client):
    r = client.get("/admin/logs")
    assert r.status_code == 403

    r = client.get("/admin/logs", headers={"x-admin-key": "wrong-key"})
    assert r.status_code == 403


def test_admin_logs_export_with_valid_key(client, valid_payload):
    client.post("/predict", json=valid_payload)  # génère au moins un évènement

    r = client.get("/admin/logs", headers={"x-admin-key": "changeme"})
    assert r.status_code == 200
    assert len(r.text.strip().splitlines()) >= 1
    # Chaque ligne doit être un JSON valide
    for line in r.text.strip().splitlines():
        json.loads(line)


def test_admin_logs_stats_requires_valid_key(client):
    r = client.get("/admin/logs/stats")
    assert r.status_code == 403


def test_admin_logs_stats_with_valid_key(client, valid_payload):
    client.post("/predict", json=valid_payload)

    r = client.get("/admin/logs/stats", headers={"x-admin-key": "changeme"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_events"] >= 1
    assert "n_errors" in body
