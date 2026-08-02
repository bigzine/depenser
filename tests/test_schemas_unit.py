"""
Tests UNITAIRES : validation du schéma Pydantic `ClientFeatures`, sans passer
par la couche HTTP/FastAPI. Ces tests vérifient la logique de validation
elle-même (champs obligatoires, plages de valeurs, types).
"""

import pytest
from pydantic import ValidationError

from api.schemas import ClientFeatures


def test_valid_features_pass():
    """Un jeu de features valide ne doit lever aucune erreur."""
    features = ClientFeatures(EXT_SOURCE_MEAN=0.5, AMT_CREDIT=100_000)
    assert features.EXT_SOURCE_MEAN == 0.5
    assert features.AMT_CREDIT == 100_000


def test_missing_ext_source_mean_rejected():
    """EXT_SOURCE_MEAN est obligatoire : son absence doit être rejetée."""
    with pytest.raises(ValidationError) as exc_info:
        ClientFeatures(AMT_CREDIT=100_000)
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("EXT_SOURCE_MEAN",) and e["type"] == "missing" for e in errors)


def test_missing_amt_credit_rejected():
    """AMT_CREDIT est obligatoire : son absence doit être rejetée."""
    with pytest.raises(ValidationError) as exc_info:
        ClientFeatures(EXT_SOURCE_MEAN=0.5)
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("AMT_CREDIT",) and e["type"] == "missing" for e in errors)


@pytest.mark.parametrize("age", [-5, 0, 17, 150])
def test_age_out_of_range_rejected(age):
    """AGE_YEARS doit être compris entre 18 et 100."""
    with pytest.raises(ValidationError):
        ClientFeatures(EXT_SOURCE_MEAN=0.5, AMT_CREDIT=100_000, AGE_YEARS=age)


@pytest.mark.parametrize("amt_credit", [0, -100, -1_000_000])
def test_negative_or_zero_credit_rejected(amt_credit):
    """AMT_CREDIT doit être strictement positif (un crédit de 0 ou négatif n'a pas de sens)."""
    with pytest.raises(ValidationError):
        ClientFeatures(EXT_SOURCE_MEAN=0.5, AMT_CREDIT=amt_credit)


@pytest.mark.parametrize("ext_source", [-0.1, 1.5, 2.0])
def test_ext_source_out_of_bounds_rejected(ext_source):
    """Les scores EXT_SOURCE_* sont normalisés entre 0 et 1."""
    with pytest.raises(ValidationError):
        ClientFeatures(EXT_SOURCE_MEAN=ext_source, AMT_CREDIT=100_000)


@pytest.mark.parametrize("rating", [0, 4, -1])
def test_region_rating_out_of_range_rejected(rating):
    """REGION_RATING_CLIENT ne prend que les valeurs 1, 2 ou 3."""
    with pytest.raises(ValidationError):
        ClientFeatures(
            EXT_SOURCE_MEAN=0.5, AMT_CREDIT=100_000, REGION_RATING_CLIENT=rating
        )


def test_wrong_type_string_instead_of_float_rejected():
    """Un texte à la place d'un nombre doit être rejeté."""
    with pytest.raises(ValidationError) as exc_info:
        ClientFeatures(EXT_SOURCE_MEAN=0.5, AMT_CREDIT="beaucoup_d_argent")
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("AMT_CREDIT",) for e in errors)


def test_to_feature_dict_merges_known_and_additional():
    """to_feature_dict() doit fusionner les champs nommés et additional_features."""
    features = ClientFeatures(
        EXT_SOURCE_MEAN=0.5,
        AMT_CREDIT=100_000,
        additional_features={"SOME_OTHER_FEATURE": 3.14},
    )
    result = features.to_feature_dict()
    assert result["EXT_SOURCE_MEAN"] == 0.5
    assert result["AMT_CREDIT"] == 100_000
    assert result["SOME_OTHER_FEATURE"] == 3.14


def test_to_feature_dict_excludes_unset_optional_fields():
    """Les champs optionnels non fournis ne doivent pas apparaître dans le dict final
    (ils seront traités comme NaN par le modèle, pas comme 0 ou None explicite)."""
    features = ClientFeatures(EXT_SOURCE_MEAN=0.5, AMT_CREDIT=100_000)
    result = features.to_feature_dict()
    assert "AGE_YEARS" not in result
    assert "EXT_SOURCE_2" not in result
