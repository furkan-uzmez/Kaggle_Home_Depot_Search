import numpy as np
import pytest

optuna = pytest.importorskip("optuna", reason="optuna not installed")

from home_depot_search.models.hpo import sample_params, build_hpo_pipeline


class TestHPO:
    def test_sample_params_ridge(self):
        trial = optuna.trial.FixedTrial({
            "alpha": 1.0,
            "solver": "svd",
            "max_features": 5000,
            "ngram_range_max": 2,
            "svd_components": 100,
        })
        params = sample_params(trial, "ridge", "tfidf-svd")
        assert "alpha" in params
        assert "solver" in params
        assert "max_features" in params

    def test_build_hpo_pipeline(self):
        params = {
            "alpha": 1.0,
            "solver": "svd",
            "max_features": 5000,
            "ngram_range_max": 2,
            "svd_components": 100,
        }
        vec, svd, model = build_hpo_pipeline(params, seed=42)
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("vec", vec), ("svd", svd), ("clf", model)])
        texts = ["hello world", "foo bar baz", "a b c d e"]
        pipe.fit(texts, [1.0, 2.0, 3.0])
        preds = pipe.predict(texts)
        assert len(preds) == 3
