import numpy as np
from sklearn.pipeline import Pipeline

from home_depot_search.models.text_features import (
    build_tfidf_svd_pipeline,
    compute_tfidf_svd_features,
)


def test_build_tfidf_svd_pipeline_returns_pipeline():
    pipeline = build_tfidf_svd_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_compute_tfidf_svd_features_returns_correct_shape():
    texts = [
        "this is a sample product description",
        "another product with different features",
        "rust resistant steel bracket heavy duty",
        "the quick brown fox jumps over",
        "high quality wooden shelf for storage",
        "plastic container with lid for kitchen",
        "stainless steel knife set professional",
        "organic cotton towels soft absorbent",
        "led light bulb energy efficient warm",
        "ceramic coffee mug dishwasher safe",
    ]
    svd_components = 5
    features = compute_tfidf_svd_features(texts, svd_components=svd_components)
    assert features.shape == (10, svd_components)


def test_compute_tfidf_svd_reuses_pipeline():
    texts = [
        "heavy duty steel bracket for construction",
        "soft cotton towel for kitchen use",
        "stainless steel knife with wooden handle",
        "led light bulb warm white energy efficient",
        "ceramic mug dishwasher safe microwave",
        "plastic storage container with lid",
        "organic bamboo cutting board large",
        "cast iron skillet pre seasoned",
        "nonstick frying pan induction compatible",
        "glass food storage set airtight",
    ]
    pipeline = build_tfidf_svd_pipeline(svd_components=5)
    pipeline.fit_transform(texts)
    result = compute_tfidf_svd_features(texts, pipeline=pipeline)
    assert result.shape == (10, 5)
