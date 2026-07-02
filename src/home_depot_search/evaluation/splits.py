import pathlib

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from home_depot_search.utils.reproducibility import set_reproducibility


def make_relevance_stratified_folds(train_df, n_splits=5, seed=42):
    set_reproducibility(seed)

    try:
        bins = pd.qcut(train_df["relevance"].rank(method="first"), q=5, labels=False)
    except ValueError:
        bins = pd.qcut(
            train_df["relevance"].round(2).rank(method="first"), q=5, labels=False
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_df = pd.DataFrame({"id": train_df["id"], "fold": -1})

    for fold_number, (_, valid_idx) in enumerate(skf.split(fold_df, bins)):
        fold_df.loc[valid_idx, "fold"] = fold_number

    return fold_df


def load_fold_manifest(filepath):
    fold_df = pd.read_csv(filepath)
    fold_df["id"] = fold_df["id"].astype("Int32")
    fold_df["fold"] = fold_df["fold"].astype(int)
    return fold_df


def save_fold_manifest(fold_df, filepath):
    pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(filepath, index=False)


def get_train_valid_indices(fold_df, fold_number):
    train_ids = fold_df.loc[fold_df["fold"] != fold_number, "id"]
    valid_ids = fold_df.loc[fold_df["fold"] == fold_number, "id"]
    return train_ids, valid_ids
