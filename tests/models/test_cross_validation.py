import numpy as np
import pandas as pd

from home_depot_search.models.cross_validation import (
    run_fold,
    run_cv_full,
    run_final_evaluation,
)


train_df = pd.DataFrame({
    "id": np.arange(20),
    "product_uid": np.arange(20),
    "product_title": ["title " + str(i) for i in range(20)],
    "search_term": ["search " + str(i) for i in range(20)],
    "relevance": np.random.uniform(1, 3, 20).round(2),
})

fold_df = pd.DataFrame({
    "id": np.arange(20, dtype="int32"),
    "fold": np.repeat([0, 1, 2, 3, 4], 4),
})


def dummy_feature_fn(df):
    return np.random.randn(len(df), 5)


def dummy_model_fn(X, y):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0).fit(X, y)


def test_run_fold_returns_tuple():
    _, valid_df = train_df.iloc[:12], train_df.iloc[12:]
    result = run_fold(train_df, valid_df, dummy_feature_fn, dummy_model_fn)
    assert isinstance(result, tuple)
    assert len(result) == 3
    ids, y_true, y_pred = result
    assert len(ids) == len(y_true) == len(y_pred)


def test_run_cv_full_returns_oof_and_metrics():
    oof_df, metrics = run_cv_full(train_df, fold_df, dummy_feature_fn, dummy_model_fn)
    assert isinstance(oof_df, pd.DataFrame)
    assert isinstance(metrics, dict)
    for key in ["mean_rmse", "std_rmse", "mean_clipped_rmse", "std_clipped_rmse", "fold_metrics"]:
        assert key in metrics


def test_run_final_evaluation_returns_metrics():
    oof_df, _ = run_cv_full(train_df, fold_df, dummy_feature_fn, dummy_model_fn)
    metrics = run_final_evaluation(oof_df)
    for key in [
        "mean_rmse", "std_rmse", "mean_clipped_rmse", "std_clipped_rmse",
        "overall_rmse", "overall_clipped_rmse", "fold_metrics",
    ]:
        assert key in metrics
