import numpy as np
import pandas as pd
import pytest

from home_depot_search.models.baselines import MeanRegressor
from home_depot_search.models.registry import (
    FEATURE_REGISTRY,
    get_feature_fn,
    get_model_fn,
    list_features,
    list_models,
    register_feature,
)
from sklearn.linear_model import Ridge


@pytest.fixture
def sample_df():
    texts = [f"unique word {i} for testing tfidf vectorizer" for i in range(400)]
    return pd.DataFrame({
        "product_text_raw": texts,
        "search_term_raw": texts,
        "product_title_raw": texts,
        "product_description": [f"desc {i}" for i in range(400)],
    })


def test_list_features_returns_list_of_tuples():
    result = list_features()
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2


def test_list_models_returns_list_of_tuples():
    result = list_models()
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2


def test_get_feature_fn_baseline_mean(sample_df):
    fn = get_feature_fn("baseline-mean")
    out = fn(sample_df)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(sample_df), 1)
    np.testing.assert_allclose(out, np.ones((len(sample_df), 1)))


def test_get_feature_fn_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_feature_fn("does-not-exist")


def test_get_model_fn_baseline_mean():
    model = get_model_fn("baseline-mean")
    assert isinstance(model, MeanRegressor)


def test_get_model_fn_ridge():
    model = get_model_fn("ridge")
    assert isinstance(model, Ridge)


def test_get_model_fn_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_model_fn("does-not-exist")


def test_feature_fn_tfidf_svd_output_shape(sample_df):
    fn = get_feature_fn("tfidf-svd")
    out = fn(sample_df)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(sample_df), 300)
    assert out.dtype == np.float64


def test_feature_fn_text_overlap_v2(sample_df):
    fn = get_feature_fn("text-overlap-v2")
    out = fn(sample_df)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(sample_df), 10)
    assert out.dtype == np.float64


def test_model_fn_baseline_mean_fit_predict():
    X = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = get_model_fn("baseline-mean")
    model.fit(X, y)
    preds = model.predict(X)
    np.testing.assert_allclose(preds, np.full(5, np.mean(y)))


def test_register_feature_decorator_works():
    @register_feature("test-feature")
    def _(seed=42):
        def fn(df, seed=42):
            return np.ones((len(df), 1))
        return fn, "Test feature"

    names = [name for name, _ in list_features()]
    assert "test-feature" in names

    fn = get_feature_fn("test-feature")
    out = fn(pd.DataFrame({"a": [1, 2]}))
    np.testing.assert_allclose(out, np.ones((2, 1)))

    FEATURE_REGISTRY.pop("test-feature", None)
