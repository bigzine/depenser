"""
Tests D'INTÉGRATION : requêtes HTTP complètes contre l'application FastAPI
(routing, validation Pydantic, chargement du modèle, sérialisation de la
réponse). Contrairement à test_schemas_unit.py, ces tests passent par la
couche HTTP réelle via TestClient.
"""


# ── Endpoints "meta" ──────────────────────────────────────────────────────

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "docs" in r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "test_model"
    assert body["n_features"] == 6
    assert body["seuil_optimal"] == 0.5


# ── /predict : cas nominal ────────────────────────────────────────────────

def test_predict_valid_payload_returns_200(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_default"] <= 1.0
    assert body["decision"] in ("ACCEPTE", "REFUSE")
    assert body["sk_id_curr"] == valid_payload["sk_id_curr"]
    assert body["threshold_used"] == 0.5
    assert body["inference_time_ms"] >= 0


def test_predict_decision_matches_threshold(client, valid_payload):
    """La décision doit être cohérente avec le seuil renvoyé."""
    r = client.post("/predict", json=valid_payload)
    body = r.json()
    if body["probability_default"] >= body["threshold_used"]:
        assert body["decision"] == "REFUSE"
    else:
        assert body["decision"] == "ACCEPTE"


def test_predict_reports_missing_features_count(client):
    """Avec seulement les champs obligatoires, missing_features doit compter
    les features attendues par le modèle mais non fournies."""
    r = client.post(
        "/predict",
        json={"sk_id_curr": 42, "features": {"EXT_SOURCE_MEAN": 0.5, "AMT_CREDIT": 100_000}},
    )
    assert r.status_code == 200
    # Le modèle factice attend 6 features ; seules 2 sont fournies ici.
    assert r.json()["missing_features"] == 4


def test_predict_accepts_additional_features(client):
    r = client.post(
        "/predict",
        json={
            "sk_id_curr": 7,
            "features": {
                "EXT_SOURCE_MEAN": 0.5,
                "AMT_CREDIT": 100_000,
                "additional_features": {"REGION_RATING_CLIENT": 2, "AGE_YEARS": 40},
            },
        },
    )
    assert r.status_code == 200
    assert r.json()["missing_features"] == 2  # EXT_SOURCE_2, EXT_SOURCE_3 manquants


def test_predict_without_sk_id_curr_is_optional(client):
    r = client.post(
        "/predict",
        json={"features": {"EXT_SOURCE_MEAN": 0.5, "AMT_CREDIT": 100_000}},
    )
    assert r.status_code == 200
    assert r.json()["sk_id_curr"] is None


# ── /predict : validation — champs obligatoires manquants ──────────────────

def test_predict_missing_required_field_returns_422(client):
    r = client.post("/predict", json={"sk_id_curr": 1, "features": {"AMT_CREDIT": 100_000}})
    assert r.status_code == 422
    assert any("EXT_SOURCE_MEAN" in str(e["loc"]) for e in r.json()["detail"])


def test_predict_empty_features_returns_422(client):
    r = client.post("/predict", json={"sk_id_curr": 1, "features": {}})
    assert r.status_code == 422


def test_predict_missing_features_key_entirely_returns_422(client):
    r = client.post("/predict", json={"sk_id_curr": 1})
    assert r.status_code == 422


# ── /predict : validation — valeurs hors plage ──────────────────────────────

def test_predict_negative_age_returns_422(client):
    r = client.post(
        "/predict",
        json={
            "sk_id_curr": 1,
            "features": {"EXT_SOURCE_MEAN": 0.5, "AMT_CREDIT": 100_000, "AGE_YEARS": -5},
        },
    )
    assert r.status_code == 422


def test_predict_zero_credit_returns_422(client):
    r = client.post(
        "/predict",
        json={"sk_id_curr": 1, "features": {"EXT_SOURCE_MEAN": 0.5, "AMT_CREDIT": 0}},
    )
    assert r.status_code == 422


def test_predict_ext_source_above_one_returns_422(client):
    r = client.post(
        "/predict",
        json={"sk_id_curr": 1, "features": {"EXT_SOURCE_MEAN": 1.5, "AMT_CREDIT": 100_000}},
    )
    assert r.status_code == 422


def test_predict_invalid_region_rating_returns_422(client):
    r = client.post(
        "/predict",
        json={
            "sk_id_curr": 1,
            "features": {
                "EXT_SOURCE_MEAN": 0.5,
                "AMT_CREDIT": 100_000,
                "REGION_RATING_CLIENT": 4,
            },
        },
    )
    assert r.status_code == 422


# ── /predict : validation — types incorrects ────────────────────────────────

def test_predict_string_instead_of_number_returns_422(client):
    r = client.post(
        "/predict",
        json={
            "sk_id_curr": 1,
            "features": {"EXT_SOURCE_MEAN": 0.5, "AMT_CREDIT": "pas_un_nombre"},
        },
    )
    assert r.status_code == 422


def test_predict_string_instead_of_int_for_sk_id_curr_returns_422(client):
    r = client.post(
        "/predict",
        json={
            "sk_id_curr": "abc",
            "features": {"EXT_SOURCE_MEAN": 0.5, "AMT_CREDIT": 100_000},
        },
    )
    assert r.status_code == 422
