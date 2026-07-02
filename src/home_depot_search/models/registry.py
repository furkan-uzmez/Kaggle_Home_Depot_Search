import numpy as np

from home_depot_search.models.text_features import build_tfidf_svd_pipeline

FEATURE_REGISTRY = {}


def register_feature(name):
    def decorator(factory):
        FEATURE_REGISTRY[name] = factory
        return factory
    return decorator


def get_feature_fn(name, seed=42):
    factory = FEATURE_REGISTRY[name]
    return factory(seed=seed)[0]


def list_features():
    return [(name, factory(seed=42)[1]) for name, factory in FEATURE_REGISTRY.items()]


@register_feature("baseline-mean")
def _(seed=42):
    def feature_fn(df, seed=42):
        return np.ones((len(df), 1))
    return feature_fn, "Mean baseline"


@register_feature("tfidf-svd")
def _(seed=42):
    _pipeline = None
    def feature_fn(df, seed=42):
        nonlocal _pipeline
        if _pipeline is None:
            _pipeline = build_tfidf_svd_pipeline(seed=seed)
            return _pipeline.fit_transform(df['product_text_raw']).astype(np.float64)
        return _pipeline.transform(df['product_text_raw']).astype(np.float64)
    return feature_fn, "TF-IDF + SVD (300 dim)"


@register_feature("tfidf-svd-search-term")
def _(seed=42):
    _pipeline = None
    def feature_fn(df, seed=42):
        nonlocal _pipeline
        if _pipeline is None:
            _pipeline = build_tfidf_svd_pipeline(seed=seed)
            return _pipeline.fit_transform(df['search_term_raw']).astype(np.float64)
        return _pipeline.transform(df['search_term_raw']).astype(np.float64)
    return feature_fn, "TF-IDF + SVD on search terms (300 dim)"


@register_feature("tfidf-svd-product-title")
def _(seed=42):
    _pipeline = None
    def feature_fn(df, seed=42):
        nonlocal _pipeline
        if _pipeline is None:
            _pipeline = build_tfidf_svd_pipeline(seed=seed)
            return _pipeline.fit_transform(df['product_title_raw']).astype(np.float64)
        return _pipeline.transform(df['product_title_raw']).astype(np.float64)
    return feature_fn, "TF-IDF + SVD on product titles (300 dim)"


@register_feature("tfidf-svd-all-text")
def _(seed=42):
    _pipeline = None
    def feature_fn(df, seed=42):
        nonlocal _pipeline
        if _pipeline is None:
            _pipeline = build_tfidf_svd_pipeline(seed=seed)
            return _pipeline.fit_transform(df['product_text_raw']).astype(np.float64)
        return _pipeline.transform(df['product_text_raw']).astype(np.float64)
    return feature_fn, "TF-IDF + SVD on full product text (300 dim)"


@register_feature("text-overlap-v2")
def _(seed=42):
    _scaler = None
    def feature_fn(df, seed=42):
        nonlocal _scaler
        search_len = df["search_term_raw"].str.len().values.astype(np.float64)
        title_len = df["product_title_raw"].str.len().values.astype(np.float64)
        desc_len = df["product_description"].str.len().values.astype(np.float64)
        product_len = df["product_text_raw"].str.len().values.astype(np.float64)

        search_words = df["search_term_raw"].str.split()
        title_words = df["product_title_raw"].str.split()
        desc_words = df["product_description"].str.split()

        def word_overlap(query_words, target_words):
            q_set = set(query_words)
            t_set = set(target_words)
            return len(q_set & t_set) / max(len(q_set | t_set), 1)

        overlap_title = np.array([
            word_overlap(q, t) for q, t in zip(search_words, title_words)
        ], dtype=np.float64)
        overlap_desc = np.array([
            word_overlap(q, d) for q, d in zip(search_words, desc_words)
        ], dtype=np.float64)

        brand_keywords = [
            "simpson", "behr", "dewalt", "milwaukee", "bosch", "makita",
            "stanley", "black+decker", "ryobi", "craftsman", "kohler",
            "moen", "lg", "samsung", "ge", "whirlpool",
        ]
        title_lower = df["product_title_raw"].str.lower()
        has_brand = np.array([
            any(b in t for b in brand_keywords)
            for t in title_lower
        ], dtype=np.float64).reshape(-1, 1)

        features = np.column_stack([
            search_len,
            title_len,
            desc_len,
            product_len,
            search_len / (title_len + 1),
            desc_len / (product_len + 1),
            overlap_title,
            overlap_desc,
            overlap_title - overlap_desc,
            has_brand[:, 0],
        ])
        if _scaler is None:
            _scaler = StandardScaler()
            return _scaler.fit_transform(features)
        return _scaler.transform(features)
    return feature_fn, "Text overlap v2: lengths, ratios, Jaccard overlap, brand match (10 dim)"


from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from home_depot_search.models.baselines import MeanRegressor, MedianRegressor

MODEL_REGISTRY = {}


def register_model(name):
    def decorator(factory):
        MODEL_REGISTRY[name] = factory
        return factory
    return decorator


def get_model_fn(name, seed=42):
    factory = MODEL_REGISTRY[name]
    return factory(seed=seed)


def list_models():
    return [(name, factory(seed=42).__class__.__name__) for name, factory in MODEL_REGISTRY.items()]


@register_model("baseline-mean")
def _(seed=42):
    return MeanRegressor()


@register_model("baseline-median")
def _(seed=42):
    return MedianRegressor()


@register_model("ridge")
def _(seed=42):
    return Ridge(alpha=1.0, random_state=seed)
