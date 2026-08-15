"""
Tests UNITAIRES pour ScoringModel._build_row (construction numpy, étape 4 —
optimisation performance). Vérifie que la construction directe en numpy
(sans pandas) produit un tableau correctement formé et dans le bon ordre.
"""

import numpy as np


def test_build_row_returns_numpy_array(client):
    from api.model_loader import scoring_model

    row = scoring_model._build_row({"EXT_SOURCE_MEAN": 0.5, "AMT_CREDIT": 100_000})
    assert isinstance(row, np.ndarray)
    assert row.shape == (1, len(scoring_model.features))
    assert row.dtype == np.float64


def test_build_row_places_values_at_correct_index(client):
    from api.model_loader import scoring_model

    features = {"EXT_SOURCE_MEAN": 0.42}
    row = scoring_model._build_row(features)
    idx = scoring_model._feature_index["EXT_SOURCE_MEAN"]
    assert row[0, idx] == 0.42


def test_build_row_fills_missing_with_nan(client):
    from api.model_loader import scoring_model

    row = scoring_model._build_row({"EXT_SOURCE_MEAN": 0.5})
    idx_missing = scoring_model._feature_index["AMT_CREDIT"]
    assert np.isnan(row[0, idx_missing])


def test_build_row_ignores_unknown_features(client):
    from api.model_loader import scoring_model

    # Ne doit pas lever d'erreur, ni modifier la forme attendue.
    row = scoring_model._build_row({"EXT_SOURCE_MEAN": 0.5, "UNE_FEATURE_INCONNUE": 999})
    assert row.shape == (1, len(scoring_model.features))


def test_predict_proba_consistent_across_calls(client, valid_payload):
    """La prédiction doit être déterministe (même entrée -> même sortie)."""
    from api.model_loader import scoring_model

    features = valid_payload["features"]
    p1 = scoring_model.predict_proba(features)
    p2 = scoring_model.predict_proba(features)
    assert p1 == p2
