"""Audit-artifact generation for the final model audit notebook.

Builds the persisted evidence consumed by ``notebooks/03_final_model_audit``:
a labeled holdout prediction log, paired robustness/stress/shortcut probe
predictions, and title-token deletion attributions. All perturbations either
flow into ``product_text_raw`` (the only field the classical candidate reads)
or deliberately target the query fields the candidate ignores, so the audit
can expose that reliance honestly.
"""

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from home_depot_search.evaluation.splits import make_relevance_stratified_folds

PredictFn = Callable[[pd.DataFrame], np.ndarray]


def rebuild_product_text_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompose ``product_text_raw`` exactly as ``build_merged_dataset`` does."""
    rebuilt = frame.copy()
    rebuilt["product_text_raw"] = (
        rebuilt["product_title_raw"]
        + " "
        + rebuilt["product_description"]
        + " "
        + rebuilt["attribute_text_raw"]
    ).str.strip()
    return rebuilt


def make_holdout_split(
    train_df: pd.DataFrame,
    seed: int = 42,
    n_splits: int = 5,
    holdout_fold: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off one relevance-stratified fold as the audit holdout."""
    fold_df = make_relevance_stratified_folds(train_df, n_splits=n_splits, seed=seed)
    holdout_ids = fold_df.loc[fold_df["fold"].eq(holdout_fold), "id"]
    holdout_mask = train_df["id"].isin(holdout_ids)
    return train_df.loc[~holdout_mask].copy(), train_df.loc[holdout_mask].copy()


def swap_adjacent_characters(text: str) -> str:
    """Swap the two middle characters; deterministic single-typo perturbation."""
    if len(text) < 2:
        return text
    position = max((len(text) // 2) - 1, 0)
    return (
        text[:position]
        + text[position + 1]
        + text[position]
        + text[position + 2 :]
    )


def _uppercase_title(holdout: pd.DataFrame) -> pd.DataFrame:
    perturbed = holdout.copy()
    perturbed["product_title_raw"] = perturbed["product_title_raw"].str.upper()
    return rebuild_product_text_raw(perturbed)


def _typo_title(holdout: pd.DataFrame) -> pd.DataFrame:
    perturbed = holdout.copy()
    perturbed["product_title_raw"] = perturbed["product_title_raw"].map(
        swap_adjacent_characters
    )
    return rebuild_product_text_raw(perturbed)


def _doubled_whitespace_description(holdout: pd.DataFrame) -> pd.DataFrame:
    perturbed = holdout.copy()
    perturbed["product_description"] = perturbed["product_description"].str.replace(
        " ", "  ", regex=False
    )
    return rebuild_product_text_raw(perturbed)


def _empty_description(holdout: pd.DataFrame) -> pd.DataFrame:
    perturbed = holdout.copy()
    perturbed["product_description"] = ""
    return rebuild_product_text_raw(perturbed)


def _tiled_long_description(holdout: pd.DataFrame) -> pd.DataFrame:
    perturbed = holdout.copy()
    perturbed["product_description"] = (
        (perturbed["product_description"] + " ") * 5
    ).str.strip()
    return rebuild_product_text_raw(perturbed)


def _shuffled_query_tokens(holdout: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    def shuffle_tokens(text: str) -> str:
        tokens = text.split()
        rng.shuffle(tokens)
        return " ".join(tokens)

    perturbed = holdout.copy()
    perturbed["search_term_raw"] = perturbed["search_term_raw"].map(shuffle_tokens)
    perturbed["search_term"] = perturbed["search_term_raw"]
    return perturbed


def _mismatched_query(holdout: pd.DataFrame) -> pd.DataFrame:
    perturbed = holdout.copy()
    perturbed["search_term_raw"] = np.roll(
        perturbed["search_term_raw"].to_numpy(), 1
    )
    perturbed["search_term"] = np.roll(perturbed["search_term"].to_numpy(), 1)
    return perturbed


PROBE_SCENARIOS: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "uppercase_title": _uppercase_title,
    "typo_title": _typo_title,
    "doubled_whitespace_description": _doubled_whitespace_description,
}

STRESS_SCENARIOS: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "empty_description": _empty_description,
    "tiled_long_description": _tiled_long_description,
}

SHORTCUT_SCENARIOS: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "shuffled_query_tokens": _shuffled_query_tokens,
    "mismatched_query": _mismatched_query,
}


def build_paired_probe_frame(
    holdout: pd.DataFrame,
    baseline_predictions: np.ndarray,
    scenarios: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
    predict_fn: PredictFn,
) -> pd.DataFrame:
    """Run each scenario and pair its predictions with the baseline by id."""
    paired_frames = []
    for scenario_name, scenario_fn in scenarios.items():
        perturbed = scenario_fn(holdout)
        paired_frames.append(
            pd.DataFrame(
                {
                    "id": holdout["id"].to_numpy(),
                    "scenario": scenario_name,
                    "baseline_prediction": baseline_predictions,
                    "probe_prediction": predict_fn(perturbed),
                }
            )
        )
    return pd.concat(paired_frames, ignore_index=True)


def compute_title_deletion_attributions(
    holdout: pd.DataFrame,
    predict_fn: PredictFn,
    sample_size: int = 200,
    seed: int = 42,
    max_tokens: int = 20,
) -> pd.DataFrame:
    """Deletion-based attribution over title tokens.

    For each sampled row, every title token (up to ``max_tokens``) is deleted
    in turn; the token whose deletion moves the prediction most is reported
    with ``faithfulness_deletion_delta = baseline - deleted``.
    """
    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, len(holdout))
    sampled = holdout.iloc[
        rng.choice(len(holdout), size=sample_size, replace=False)
    ].copy()
    baseline_predictions = predict_fn(sampled)

    variant_rows = []
    for sample_position, (_, row) in enumerate(sampled.iterrows()):
        tokens = str(row["product_title_raw"]).split()[:max_tokens]
        for token_index, token in enumerate(tokens):
            remaining_tokens = tokens[:token_index] + tokens[token_index + 1 :]
            variant = row.copy()
            variant["product_title_raw"] = " ".join(remaining_tokens)
            variant_rows.append(
                {
                    "sample_position": sample_position,
                    "deleted_token": token,
                    **variant.to_dict(),
                }
            )
    variants = rebuild_product_text_raw(pd.DataFrame(variant_rows))
    variants["deleted_prediction"] = predict_fn(variants)
    variants["baseline_prediction"] = baseline_predictions[
        variants["sample_position"].to_numpy()
    ]
    variants["faithfulness_deletion_delta"] = (
        variants["baseline_prediction"] - variants["deleted_prediction"]
    )
    top_variant_indices = (
        variants["faithfulness_deletion_delta"]
        .abs()
        .groupby(variants["sample_position"])
        .idxmax()
    )
    top_variants = variants.loc[top_variant_indices]
    return top_variants[
        [
            "id",
            "deleted_token",
            "baseline_prediction",
            "deleted_prediction",
            "faithfulness_deletion_delta",
        ]
    ].reset_index(drop=True)


def update_artifact_manifest(
    manifest_path: Path | str,
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Merge ``records`` into the manifest JSON, preserving existing entries."""
    manifest_path = Path(manifest_path)
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            manifest = loaded
    manifest.update({name: dict(record) for name, record in records.items()})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=4) + "\n", encoding="utf-8"
    )
    return manifest
