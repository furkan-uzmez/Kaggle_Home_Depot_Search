import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def clipped_rmse(y_true, y_pred, lower=1.0, upper=3.0):
    y_pred_clipped = np.clip(np.asarray(y_pred), lower, upper)
    return float(np.sqrt(np.mean((np.asarray(y_true) - y_pred_clipped) ** 2)))
