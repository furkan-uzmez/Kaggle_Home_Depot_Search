#!/usr/bin/env python3
"""Generate the persisted audit artifacts for notebooks/03_final_model_audit.

Fits the frozen classical candidate (tfidf-svd-all-text + ridge) on the
non-holdout folds, evaluates it on a relevance-stratified holdout, and
persists: prediction_log.csv, probe/stress/shortcut paired predictions,
title-deletion attributions, and matching artifact-manifest records.
No network access; only local data and registry code are used.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from home_depot_search.audit import (
    PROBE_SCENARIOS,
    SHORTCUT_SCENARIOS,
    STRESS_SCENARIOS,
    build_paired_probe_frame,
    compute_title_deletion_attributions,
    make_holdout_split,
    update_artifact_manifest,
)
from home_depot_search.data.data_loader import build_merged_dataset
from home_depot_search.models.registry import get_feature_fn, get_model_fn
from home_depot_search.models.text_features import build_tfidf_svd_pipeline
from home_depot_search.utils.reproducibility import set_reproducibility

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "tfidf-svd-all-text + ridge"


def short_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--attribution-sample", type=int, default=200)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run on a 3000-row subsample for a fast end-to-end check",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / "data"

    train_df, _ = build_merged_dataset(
        data_dir / "train.csv",
        data_dir / "test.csv",
        data_dir / "product_descriptions.csv",
        data_dir / "attributes.csv",
    )
    if args.smoke:
        train_df = train_df.sample(n=3000, random_state=args.seed).reset_index(
            drop=True
        )

    fit_df, holdout_df = make_holdout_split(train_df, seed=args.seed)
    print(f"fit rows: {len(fit_df)}, holdout rows: {len(holdout_df)}")

    set_reproducibility(args.seed)
    feature_fn = get_feature_fn("tfidf-svd-all-text", seed=args.seed)
    model = get_model_fn("ridge", seed=args.seed)
    model.fit(feature_fn(fit_df), fit_df["relevance"].to_numpy(dtype=float))

    def predict_fn(frame: pd.DataFrame) -> np.ndarray:
        return model.predict(feature_fn(frame))

    baseline_predictions = predict_fn(holdout_df)
    holdout_rmse = float(
        np.sqrt(
            np.mean(
                (holdout_df["relevance"].to_numpy(dtype=float) - baseline_predictions)
                ** 2
            )
        )
    )
    print(f"holdout rmse: {holdout_rmse:.4f}")

    pipeline_params = repr(
        sorted(build_tfidf_svd_pipeline(seed=args.seed).get_params().items())
    )
    identity = {
        "candidate_id": f"tfidf-svd-all-text_ridge_seed{args.seed}",
        "tokenizer_id": "tfidf-svd-" + short_hash(pipeline_params.encode()),
        "checkpoint_id": "ridge-"
        + short_hash(model.coef_.tobytes() + np.float64(model.intercept_).tobytes()),
    }
    print(f"identity: {identity}")

    prediction_log = pd.DataFrame(
        {
            "id": holdout_df["id"].to_numpy(),
            "split": "validation",
            "y_true": holdout_df["relevance"].to_numpy(dtype=float),
            "y_pred": baseline_predictions,
            "search_term": holdout_df["search_term"].to_numpy(),
            "product_title": holdout_df["product_title"].to_numpy(),
            "product_description": holdout_df["product_description"]
            .fillna("")
            .to_numpy(),
            **identity,
        }
    )
    prediction_log.to_csv(results_dir / "prediction_log.csv", index=False)

    scenario_outputs = {
        "probe_predictions.csv": PROBE_SCENARIOS,
        "stress_probe_predictions.csv": STRESS_SCENARIOS,
        "shortcut_probe_predictions.csv": SHORTCUT_SCENARIOS,
    }
    for file_name, scenarios in scenario_outputs.items():
        paired = build_paired_probe_frame(
            holdout_df, baseline_predictions, scenarios, predict_fn
        )
        paired.to_csv(results_dir / file_name, index=False)
        print(f"{file_name}: {len(paired)} rows, scenarios={sorted(scenarios)}")

    attributions = compute_title_deletion_attributions(
        holdout_df,
        predict_fn,
        sample_size=args.attribution_sample,
        seed=args.seed,
    )
    attributions = attributions.assign(**identity)
    attributions.to_csv(results_dir / "stored_attributions.csv", index=False)
    print(f"stored_attributions.csv: {len(attributions)} rows")

    generated_files = [
        "prediction_log.csv",
        "probe_predictions.csv",
        "stress_probe_predictions.csv",
        "shortcut_probe_predictions.csv",
        "stored_attributions.csv",
    ]
    update_artifact_manifest(
        results_dir / "artifact_manifest.json",
        {
            file_name: {
                "kind": file_name.removesuffix(".csv"),
                "experiment": EXPERIMENT_NAME,
                "source": f"scripts/generate_audit_artifacts.py --seed {args.seed}",
                **identity,
            }
            for file_name in generated_files
        },
    )
    print(f"manifest updated: {results_dir / 'artifact_manifest.json'}")


if __name__ == "__main__":
    main()
