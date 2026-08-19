"""
Tests D'INTÉGRATION pour la journalisation structurée des prédictions (base
de données, voir api/db.py) et les endpoints d'export (/admin/logs), utilisés
pour l'analyse de drift.
"""

import json


def test_predict_persists_event_to_database(client, valid_payload):
    from api import db

    stats_before = db.get_stats()

    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200

    stats_after = db.get_stats()
    assert stats_after["n_events"] == stats_before["n_events"] + 1


def test_predict_validation_error_does_not_reach_logging(client):
    """Une erreur 422 (validation Pydantic) se produit avant l'appel au modèle :
    elle n'est pas journalisée comme évènement de prédiction (comportement attendu,
    car aucune tentative de scoring n'a eu lieu)."""
    from api import db

    stats_before = db.get_stats()

    r = client.post("/predict", json={"sk_id_curr": 1, "features": {}})
    assert r.status_code == 422

    stats_after = db.get_stats()
    assert stats_after["n_events"] == stats_before["n_events"]


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
    # Chaque ligne doit être un JSON valide, avec les champs attendus
    for line in r.text.strip().splitlines():
        event = json.loads(line)
        assert "timestamp" in event
        assert "inputs" in event
        assert "inference_time_ms" in event


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
