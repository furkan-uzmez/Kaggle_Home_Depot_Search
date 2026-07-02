import numpy as np
from sklearn.linear_model import Ridge
from numpy.typing import ArrayLike


def train_ridge(X: ArrayLike, y: ArrayLike, alpha: float = 1.0, seed: int = 42) -> Ridge:
    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(X, y)
    return model


def predict_ridge(model: Ridge, X: ArrayLike) -> np.ndarray:
    return model.predict(X)
