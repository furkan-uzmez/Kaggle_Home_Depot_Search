import pandas as pd
import pytest

from home_depot_search.evaluation.splits import (
    get_train_valid_indices,
    load_fold_manifest,
    make_relevance_stratified_folds,
    save_fold_manifest,
)


@pytest.fixture
def sample_train_df():
    n = 100
    return pd.DataFrame({
        "id": range(n),
        "relevance": [1.0, 1.5, 2.0, 2.5, 3.0] * (n // 5),
    }).astype({"id": "Int32"})


def test_make_relevance_stratified_folds_returns_correct_columns(sample_train_df):
    fold_df = make_relevance_stratified_folds(sample_train_df)
    assert list(fold_df.columns) == ["id", "fold"]


def test_make_relevance_stratified_folds_all_folds_present(sample_train_df):
    fold_df = make_relevance_stratified_folds(sample_train_df, n_splits=5)
    for fold in range(5):
        assert fold in fold_df["fold"].values


def test_make_relevance_stratified_folds_deterministic(sample_train_df):
    fold_df_1 = make_relevance_stratified_folds(sample_train_df, seed=42)
    fold_df_2 = make_relevance_stratified_folds(sample_train_df, seed=42)
    pd.testing.assert_frame_equal(fold_df_1, fold_df_2)


def test_save_and_load_fold_manifest(sample_train_df, tmp_path):
    fold_df = make_relevance_stratified_folds(sample_train_df)
    filepath = tmp_path / "fold_manifest.csv"
    save_fold_manifest(fold_df, filepath)
    loaded = load_fold_manifest(filepath)
    pd.testing.assert_frame_equal(fold_df, loaded)


def test_get_train_valid_indices(sample_train_df):
    fold_df = make_relevance_stratified_folds(sample_train_df, n_splits=5)
    train_ids, valid_ids = get_train_valid_indices(fold_df, fold_number=0)
    assert len(train_ids) + len(valid_ids) == len(fold_df)
    assert set(train_ids).isdisjoint(set(valid_ids))
    assert all(fold_df.loc[fold_df["id"].isin(valid_ids), "fold"] == 0)
