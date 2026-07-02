import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from home_depot_search.evaluation.metrics import clipped_rmse, rmse
from home_depot_search.evaluation.splits import get_train_valid_indices
from home_depot_search.utils.reproducibility import set_reproducibility


def run_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_fn: Callable[[pd.DataFrame], np.ndarray],
    model_fn: Callable[[np.ndarray, np.ndarray], object],
    seed: int = 42,
):
    set_reproducibility(seed)
    train_features = feature_fn(train_df)
    valid_features = feature_fn(valid_df)
    model = model_fn(train_features, train_df["relevance"].values)
    y_pred = model.predict(valid_features)
    y_true = valid_df["relevance"].values
    return valid_df["id"], y_true, y_pred


def run_cv_full(
    train_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    feature_fn: Callable[[pd.DataFrame], np.ndarray],
    model_fn: Callable[[np.ndarray, np.ndarray], object],
    seed: int = 42,
):
    set_reproducibility(seed)
    n_splits = int(fold_df["fold"].max()) + 1
    all_ids, all_true, all_pred, all_folds = [], [], [], []
    fold_metrics = []

    for fold in range(n_splits):
        train_ids, valid_ids = get_train_valid_indices(fold_df, fold)
        inner_train = train_df.loc[train_df["id"].isin(train_ids)].copy()
        inner_valid = train_df.loc[train_df["id"].isin(valid_ids)].copy()
        valid_ids_series, y_true, y_pred = run_fold(
            inner_train, inner_valid, feature_fn, model_fn, seed
        )
        all_ids.extend(valid_ids_series.values)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        all_folds.extend([fold] * len(valid_ids_series))
        fold_metrics.append(
            {
                "fold": fold,
                "rmse": rmse(y_true, y_pred),
                "clipped_rmse": clipped_rmse(y_true, y_pred),
            }
        )

    oof_df = pd.DataFrame(
        {"id": all_ids, "y_true": all_true, "y_pred": all_pred, "fold": all_folds}
    )

    fold_rmses = [m["rmse"] for m in fold_metrics]
    fold_clipped = [m["clipped_rmse"] for m in fold_metrics]
    metrics = {
        "mean_rmse": float(np.mean(fold_rmses)),
        "std_rmse": float(np.std(fold_rmses)),
        "mean_clipped_rmse": float(np.mean(fold_clipped)),
        "std_clipped_rmse": float(np.std(fold_clipped)),
        "fold_metrics": fold_metrics,
    }

    return oof_df, metrics


def run_final_evaluation(
    oof_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> dict:
    y_true = oof_df["y_true"].values
    y_pred = oof_df["y_pred"].values
    overall_rmse = rmse(y_true, y_pred)
    overall_clipped = clipped_rmse(y_true, y_pred)

    fold_metrics = []
    for fold in sorted(oof_df["fold"].unique()):
        mask = oof_df["fold"] == fold
        fm = rmse(oof_df.loc[mask, "y_true"].values, oof_df.loc[mask, "y_pred"].values)
        fc = clipped_rmse(
            oof_df.loc[mask, "y_true"].values, oof_df.loc[mask, "y_pred"].values
        )
        fold_metrics.append({"fold": int(fold), "rmse": fm, "clipped_rmse": fc})

    metrics = {
        "mean_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "std_rmse": float(np.std([m["rmse"] for m in fold_metrics])),
        "mean_clipped_rmse": float(np.mean([m["clipped_rmse"] for m in fold_metrics])),
        "std_clipped_rmse": float(np.std([m["clipped_rmse"] for m in fold_metrics])),
        "overall_rmse": overall_rmse,
        "overall_clipped_rmse": overall_clipped,
        "fold_metrics": fold_metrics,
    }

    print(f"OOF RMSE:        {overall_rmse:.5f}")
    print(f"OOF Clipped RMSE: {overall_clipped:.5f}")
    print(f"Mean Fold RMSE:   {metrics['mean_rmse']:.5f} ± {metrics['std_rmse']:.5f}")

    if logger is not None:
        logger.info("Final evaluation metrics", extra={"metrics": metrics})

    return metrics
