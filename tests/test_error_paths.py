"""
Tests UNITAIRES / D'INTÉGRATION couvrant les branches d'erreur non exercées
par les tests existants : modèle non chargé, échec de prédiction, absence de
fichier de logs, comptage des erreurs dans les logs.
"""

import os

import pytest


def test_model_info_returns_404_when_no_metadata(client, monkeypatch):
    from api import main

    monkeypatch.setattr(main.scoring_model, "metadata", {})
    r = client.get("/model-info")
    assert r.status_code == 404


def test_predict_returns_503_when_model_not_loaded(client, monkeypatch, valid_payload):
    from api import main

    monkeypatch.setattr(main.scoring_model, "booster", None)  # is_loaded -> False
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 503
    monkeypatch.undo()  # sécurité supplémentaire avant les tests suivants


def test_predict_returns_500_on_internal_prediction_error(client, monkeypatch, valid_payload):
    from api import main

    def boom(_features):
        raise RuntimeError("erreur simulée du modèle")

    monkeypatch.setattr(main.scoring_model, "predict_proba", boom)
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 500
    assert "erreur de prédiction" in r.json()["detail"].lower() or "erreur simulée" in r.json()["detail"]


def test_admin_logs_returns_empty_string_when_no_log_file(client, monkeypatch, tmp_path):
    from api import main

    fake_path = str(tmp_path / "nonexistent.jsonl")
    monkeypatch.setattr(main, "PREDICTIONS_LOG_PATH", fake_path)
    r = client.get("/admin/logs", headers={"x-admin-key": "changeme"})
    assert r.status_code == 200
    assert r.text == ""


def test_admin_logs_stats_returns_zero_when_no_log_file(client, monkeypatch, tmp_path):
    from api import main

    fake_path = str(tmp_path / "nonexistent.jsonl")
    monkeypatch.setattr(main, "PREDICTIONS_LOG_PATH", fake_path)
    r = client.get("/admin/logs/stats", headers={"x-admin-key": "changeme"})
    assert r.status_code == 200
    assert r.json() == {"n_events": 0}


def test_admin_logs_stats_counts_errors_correctly(client, monkeypatch, tmp_path, valid_payload):
    from api import main

    fake_path = str(tmp_path / "counted.jsonl")
    monkeypatch.setattr(main, "PREDICTIONS_LOG_PATH", fake_path)

    # Un succès...
    client.post("/predict", json=valid_payload)

    # ...puis une erreur simulée, journalisée par la même route.
    def boom(_features):
        raise RuntimeError("erreur simulée")
    monkeypatch.setattr(main.scoring_model, "predict_proba", boom)
    client.post("/predict", json=valid_payload)

    r = client.get("/admin/logs/stats", headers={"x-admin-key": "changeme"})
    body = r.json()
    assert body["n_events"] == 2
    assert body["n_errors"] == 1
