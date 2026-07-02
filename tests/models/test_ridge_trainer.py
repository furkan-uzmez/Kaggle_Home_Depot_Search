import numpy as np
from sklearn.linear_model import Ridge

from home_depot_search.models.ridge_trainer import train_ridge, predict_ridge


def test_train_ridge_returns_model():
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    model = train_ridge(X, y)
    assert isinstance(model, Ridge)
    assert hasattr(model, "coef_")


def test_predict_ridge_returns_correct_shape():
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    model = train_ridge(X, y)
    X_new = np.random.randn(20, 5)
    preds = predict_ridge(model, X_new)
    assert preds.shape == (20,)


def test_ridge_deterministic():
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    model_1 = train_ridge(X, y, seed=42)
    model_2 = train_ridge(X, y, seed=42)
    np.testing.assert_allclose(model_1.coef_, model_2.coef_)
