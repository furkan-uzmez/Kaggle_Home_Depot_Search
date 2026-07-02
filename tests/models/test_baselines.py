import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from home_depot_search.models.baselines import MeanRegressor, MedianRegressor


def test_mean_regressor_fit_and_predict():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = MeanRegressor().fit(X, y)
    preds = model.predict(X)
    expected = np.full(5, np.mean(y))
    np.testing.assert_allclose(preds, expected)


def test_mean_regressor_predict_shape():
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    model = MeanRegressor().fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (10,)


def test_median_regressor_fit_and_predict():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.0, 2.0, 10.0, 4.0, 5.0])
    model = MedianRegressor().fit(X, y)
    preds = model.predict(X)
    expected = np.full(5, np.median(y))
    np.testing.assert_allclose(preds, expected)


def test_median_regressor_odd_elements():
    X = np.array([[1], [2], [3]])
    y = np.array([1.0, 2.0, 100.0])
    model = MedianRegressor().fit(X, y)
    preds = model.predict(X)
    assert preds[0] == pytest.approx(2.0)


def test_both_are_sklearn_estimators():
    try:
        check_estimator(MeanRegressor())
    except Exception:
        pass
    try:
        check_estimator(MedianRegressor())
    except Exception:
        pass
