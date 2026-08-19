"""
Tests UNITAIRES / D'INTÉGRATION couvrant les branches d'erreur non exercées
par les tests existants : modèle non chargé, échec de prédiction, base de
données vide, comptage des erreurs dans les logs.
"""

from sqlalchemy import create_engine


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


def test_export_and_stats_empty_on_fresh_database(monkeypatch, tmp_path):
    """Vérifie le comportement de db.export_logs_as_jsonl()/get_stats() sur
    une base fraîchement créée, sans passer par le client HTTP partagé (dont
    la base accumule déjà des évènements des autres tests de la session)."""
    from api import db

    fresh_engine = create_engine(f"sqlite:///{tmp_path}/fresh.db")
    monkeypatch.setattr(db, "engine", fresh_engine)
    db.init_db()

    assert db.get_stats() == {"n_events": 0, "n_errors": 0}
    assert db.export_logs_as_jsonl() == ""


def test_admin_logs_stats_counts_errors_correctly(client, monkeypatch, tmp_path, valid_payload):
    from api import db, main

    fresh_engine = create_engine(f"sqlite:///{tmp_path}/counted.db")
    monkeypatch.setattr(db, "engine", fresh_engine)
    db.init_db()

    # Un succès...
    client.post("/predict", json=valid_payload)

    # ...puis une erreur simulée, journalisée par la même route.
    def boom(_features):
        raise RuntimeError("erreur simulée")
    monkeypatch.setattr(main.scoring_model, "predict_proba", boom)
    client.post("/predict", json=valid_payload)

    stats = db.get_stats()
    assert stats["n_events"] == 2
    assert stats["n_errors"] == 1
