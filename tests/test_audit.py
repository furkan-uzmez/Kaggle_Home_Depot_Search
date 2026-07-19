import json

import numpy as np
import pandas as pd
import pytest

from home_depot_search.audit import (
    PROBE_SCENARIOS,
    SHORTCUT_SCENARIOS,
    STRESS_SCENARIOS,
    build_paired_probe_frame,
    compute_title_deletion_attributions,
    make_holdout_split,
    rebuild_product_text_raw,
    swap_adjacent_characters,
    update_artifact_manifest,
)


def make_toy_train_df(rows=40):
    rng = np.random.default_rng(0)
    titles = [f"cordless drill kit {i}" for i in range(rows)]
    descriptions = [f"durable description text {i}" for i in range(rows)]
    attributes = [f"brand value {i}" for i in range(rows)]
    frame = pd.DataFrame(
        {
            "id": np.arange(1, rows + 1),
            "product_uid": np.arange(1001, 1001 + rows),
            "search_term": [f"Query Term {i}" for i in range(rows)],
            "search_term_raw": [f"query term {i}" for i in range(rows)],
            "product_title": titles,
            "product_title_raw": titles,
            "product_description": descriptions,
            "attribute_text_raw": attributes,
            "relevance": rng.uniform(1.0, 3.0, rows).round(2),
        }
    )
    return rebuild_product_text_raw(frame)


def test_rebuild_product_text_raw_matches_data_loader_composition():
    frame = pd.DataFrame(
        {
            "product_title_raw": ["drill kit"],
            "product_description": ["long description"],
            "attribute_text_raw": [""],
        }
    )
    rebuilt = rebuild_product_text_raw(frame)
    assert rebuilt.loc[0, "product_text_raw"] == "drill kit long description"


def test_make_holdout_split_is_disjoint_complete_and_deterministic():
    train_df = make_toy_train_df()
    fit_first, holdout_first = make_holdout_split(train_df, seed=42)
    fit_second, holdout_second = make_holdout_split(train_df, seed=42)
    assert set(fit_first["id"]).isdisjoint(set(holdout_first["id"]))
    assert set(fit_first["id"]) | set(holdout_first["id"]) == set(train_df["id"])
    assert holdout_first["id"].tolist() == holdout_second["id"].tolist()
    assert fit_first["id"].tolist() == fit_second["id"].tolist()
    assert len(holdout_first) == pytest.approx(len(train_df) / 5, abs=2)


def test_swap_adjacent_characters_is_deterministic_and_length_preserving():
    assert swap_adjacent_characters("drill") == swap_adjacent_characters("drill")
    assert len(swap_adjacent_characters("cordless")) == len("cordless")
    assert swap_adjacent_characters("cordless") != "cordless"
    assert swap_adjacent_characters("a") == "a"
    assert swap_adjacent_characters("") == ""


def test_probe_scenarios_perturb_only_model_visible_text():
    holdout = make_toy_train_df()
    for scenario_name, scenario_fn in {**PROBE_SCENARIOS, **STRESS_SCENARIOS}.items():
        perturbed = scenario_fn(holdout)
        assert not perturbed["product_text_raw"].equals(
            holdout["product_text_raw"]
        ), scenario_name
        assert perturbed["id"].tolist() == holdout["id"].tolist()


def test_empty_description_scenario_removes_description_text():
    holdout = make_toy_train_df()
    perturbed = STRESS_SCENARIOS["empty_description"](holdout)
    assert not perturbed["product_text_raw"].str.contains("durable description").any()


def test_shortcut_scenarios_change_query_but_not_product_text():
    holdout = make_toy_train_df()
    mismatched = SHORTCUT_SCENARIOS["mismatched_query"](holdout)
    assert not mismatched["search_term_raw"].eq(holdout["search_term_raw"]).any()
    assert mismatched["product_text_raw"].equals(holdout["product_text_raw"])

    shuffled = SHORTCUT_SCENARIOS["shuffled_query_tokens"](holdout)
    assert shuffled["product_text_raw"].equals(holdout["product_text_raw"])
    original_tokens = holdout["search_term_raw"].str.split().map(sorted)
    shuffled_tokens = shuffled["search_term_raw"].str.split().map(sorted)
    assert original_tokens.equals(shuffled_tokens)


def test_build_paired_probe_frame_pairs_baseline_and_probe_rows():
    holdout = make_toy_train_df()

    def predict_fn(frame):
        return frame["product_text_raw"].str.len().to_numpy(dtype=float)

    baseline_predictions = predict_fn(holdout)
    scenarios = {"empty_description": STRESS_SCENARIOS["empty_description"]}
    paired = build_paired_probe_frame(
        holdout, baseline_predictions, scenarios, predict_fn
    )
    assert list(paired.columns) == [
        "id",
        "scenario",
        "baseline_prediction",
        "probe_prediction",
    ]
    assert len(paired) == len(holdout)
    assert paired["scenario"].eq("empty_description").all()
    assert (paired["probe_prediction"] < paired["baseline_prediction"]).all()


def test_compute_title_deletion_attributions_reports_positive_delta_for_length_model():
    holdout = make_toy_train_df(rows=6)

    def predict_fn(frame):
        return frame["product_text_raw"].str.len().to_numpy(dtype=float)

    attributions = compute_title_deletion_attributions(
        holdout, predict_fn, sample_size=4, seed=42
    )
    assert list(attributions.columns) == [
        "id",
        "deleted_token",
        "baseline_prediction",
        "deleted_prediction",
        "faithfulness_deletion_delta",
    ]
    assert len(attributions) == 4
    assert attributions["id"].isin(holdout["id"]).all()
    assert (attributions["faithfulness_deletion_delta"] > 0).all()


def test_update_artifact_manifest_merges_without_dropping_existing_records(tmp_path):
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps({"existing.csv": {"kind": "oof", "experiment": "old"}}),
        encoding="utf-8",
    )
    update_artifact_manifest(
        manifest_path,
        {"prediction_log.csv": {"kind": "prediction_log", "experiment": "new"}},
    )
    merged = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert merged["existing.csv"] == {"kind": "oof", "experiment": "old"}
    assert merged["prediction_log.csv"] == {
        "kind": "prediction_log",
        "experiment": "new",
    }
