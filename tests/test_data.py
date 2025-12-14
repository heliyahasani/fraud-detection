"""Tests for data loading and preprocessing."""

import numpy as np

from src.data import DataLoader


def test_split_stratified(sample_data):
    """Test that train/test split maintains fraud ratio."""
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.split(sample_data)

    original_fraud_rate = sample_data["is_fraud"].mean()
    train_fraud_rate = y_train.mean()
    test_fraud_rate = y_test.mean()

    # Fraud rate should be similar across splits (within 2%)
    assert abs(train_fraud_rate - original_fraud_rate) < 0.02
    assert abs(test_fraud_rate - original_fraud_rate) < 0.02


def test_split_sizes(sample_data):
    """Test that split produces correct proportions."""
    loader = DataLoader(test_size=0.2)
    X_train, X_test, y_train, y_test = loader.split(sample_data)

    total = len(sample_data)
    expected_test = int(total * 0.2)

    assert len(X_test) == expected_test
    assert len(X_train) == total - expected_test
    assert len(y_test) == len(X_test)
    assert len(y_train) == len(X_train)


def test_smote_increases_minority(sample_X_y):
    """Test that SMOTE increases minority class samples."""
    X, y = sample_X_y
    loader = DataLoader()

    original_fraud_count = y.sum()
    X_resampled, y_resampled = loader.apply_smote(X, y, sampling_strategy=0.5)

    new_fraud_count = y_resampled.sum()
    assert new_fraud_count > original_fraud_count
    assert len(X_resampled) == len(y_resampled)
