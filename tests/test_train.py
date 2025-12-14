"""Tests for model training."""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.train import ModelType, TrainConfig, Trainer


def test_model_types_exist():
    """Test all expected model types are defined."""
    expected = {"logistic", "random_forest", "xgboost", "neural_net"}
    actual = {m.value for m in ModelType}
    assert actual == expected


def test_config_defaults():
    """Test TrainConfig has sensible defaults."""
    config = TrainConfig()

    assert config.model_type == ModelType.XGBOOST
    assert config.cv_folds == 5
    assert config.random_state == 42
    assert config.use_class_weight is True


@pytest.mark.parametrize("model_type,expected_class", [
    (ModelType.LOGISTIC, LogisticRegression),
    (ModelType.RANDOM_FOREST, RandomForestClassifier),
    (ModelType.XGBOOST, XGBClassifier),
    (ModelType.NEURAL_NET, Pipeline),
])
def test_get_model_returns_correct_type(model_type, expected_class):
    """Test that get_model returns correct model type."""
    config = TrainConfig(model_type=model_type, use_gpu=False)
    trainer = Trainer(config)
    model = trainer.get_model()

    assert isinstance(model, expected_class)


def test_cross_validate_returns_scores(sample_X_y):
    """Test cross_validate returns expected structure."""
    X, y = sample_X_y
    config = TrainConfig(model_type=ModelType.LOGISTIC, cv_folds=3)
    trainer = Trainer(config)

    result = trainer.cross_validate(X, y, scoring="f1")

    assert "mean" in result
    assert "std" in result
    assert "scores" in result
    assert len(result["scores"]) == 3
    assert 0 <= result["mean"] <= 1
