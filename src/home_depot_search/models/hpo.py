from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge


def _check_optuna():
    try:
        import optuna
        return optuna
    except ImportError:
        raise ImportError("optuna is required for HPO. Install with: uv add optuna")


def sample_params(trial, model_type, feature_type):
    params = {}
    if model_type == "ridge":
        params["alpha"] = trial.log_float("alpha", 1e-3, 1e2, log=True)
        params["solver"] = trial.suggest_categorical("solver", ["svd", "lsqr", "saga"])
    if feature_type == "tfidf-svd":
        params["max_features"] = trial.suggest_int("max_features", 1000, 20000, log=True)
        params["ngram_range_max"] = trial.suggest_int("ngram_range_max", 1, 3)
        params["svd_components"] = trial.suggest_int("svd_components", 50, 500, log=True)
    return params


def build_hpo_pipeline(params, seed=42):
    vec = TfidfVectorizer(
        max_features=int(params.get("max_features", 5000)),
        ngram_range=(1, int(params.get("ngram_range_max", 2))),
        sublinear_tf=True,
    )
    svd = TruncatedSVD(
        n_components=int(params.get("svd_components", 300)),
        random_state=seed,
    )
    model = Ridge(
        alpha=params.get("alpha", 1.0),
        solver=params.get("solver", "svd"),
        random_state=seed,
    )
    return vec, svd, model


def objective_factory(X_text, y, n_folds=3, seed=42):
    import optuna
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import Pipeline

    def objective(trial):
        params = sample_params(trial, "ridge", "tfidf-svd")
        vec, svd, model = build_hpo_pipeline(params, seed=seed)
        pipe = Pipeline([
            ("vec", vec),
            ("svd", svd),
            ("clf", model),
        ])

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []
        for train_idx, val_idx in kf.split(X_text):
            X_tr, X_val = X_text.iloc[train_idx], X_text.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_val)
            scores.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        return float(np.mean(scores))

    return objective
