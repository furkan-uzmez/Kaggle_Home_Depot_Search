from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import numpy as np


def build_tfidf_svd_pipeline(max_features=5000, svd_components=300, seed=42):
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, sublinear_tf=True, stop_words='english')),
        ('svd', TruncatedSVD(n_components=svd_components, random_state=seed)),
    ])


def compute_tfidf_svd_features(text_series, pipeline=None, max_features=5000, svd_components=300, seed=42):
    if pipeline is None:
        pipeline = build_tfidf_svd_pipeline(max_features=max_features, svd_components=svd_components, seed=seed)
        return pipeline.fit_transform(text_series).astype(np.float64)
    return pipeline.transform(text_series).astype(np.float64)
