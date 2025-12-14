"""Tests for model evaluation."""

from src.evaluate import Evaluator, Metrics


def test_metrics_dataclass():
    """Test Metrics dataclass has expected fields."""
    metrics = Metrics(
        accuracy=0.95,
        precision=0.7,
        recall=0.8,
        f1=0.75,
        roc_auc=0.85,
        pr_auc=0.8,
        threshold=0.5,
    )

    assert metrics.accuracy == 0.95
    assert metrics.pr_auc == 0.8
    assert metrics.roc_auc == 0.85
    assert metrics.f1 == 0.75
    assert metrics.precision == 0.7
    assert metrics.recall == 0.8
    assert metrics.threshold == 0.5


def test_find_best_threshold(trained_model, sample_X_y):
    """Test threshold optimization returns valid value."""
    X, y = sample_X_y
    evaluator = Evaluator(trained_model)

    threshold = evaluator.find_best_threshold(X, y, optimize_for="f1")

    assert 0 < threshold < 1


def test_evaluate_returns_metrics(trained_model, sample_X_y):
    """Test evaluate returns Metrics object with valid values."""
    X, y = sample_X_y
    evaluator = Evaluator(trained_model)

    metrics = evaluator.evaluate(X, y)

    assert isinstance(metrics, Metrics)
    assert 0 <= metrics.pr_auc <= 1
    assert 0 <= metrics.roc_auc <= 1
    assert 0 <= metrics.f1 <= 1
    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1
