"""Pytest fixtures for fraud detection tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def sample_data():
    """Create small synthetic dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_fraud = 50  # 5% fraud rate

    data = {
        "amt": np.random.exponential(100, n_samples),
        "age": np.random.randint(18, 80, n_samples),
        "hour": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "category_grocery": np.random.randint(0, 2, n_samples),
        "category_online": np.random.randint(0, 2, n_samples),
        "is_fraud": [1] * n_fraud + [0] * (n_samples - n_fraud),
    }

    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.fixture
def sample_X_y(sample_data):
    """Split sample data into X and y."""
    X = sample_data.drop(columns=["is_fraud"])
    y = sample_data["is_fraud"]
    return X, y


@pytest.fixture
def trained_model(sample_X_y):
    """Simple trained model for evaluation tests."""
    X, y = sample_X_y
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    return model
