import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class MeanRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n_samples, self.mean_)


class MedianRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.median_ = np.median(y)
        return self

    def predict(self, X):
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n_samples, self.median_)
