# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-autoscroll,-collapsed,-scrolled,-trusted,-ExecuteTime
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Final Model Audit
#
# Review persisted, labeled evidence only. This notebook never trains, loads a
# model, generates predictions, or accesses the network.

# %% tags=["parameters"]
# ruff: noqa: E402, E501
# pyright: reportUnusedExpression=false
prediction_log_path = None
artifact_manifest_path = None
probe_predictions_path = None
stress_probe_predictions_path = None
shortcut_probe_predictions_path = None
stored_attributions_path = None
candidate_id = None
tokenizer_id = None
checkpoint_id = None
evaluation_split = "validation"

# %% [markdown]
# ## Setup and Artifact Inventory
#
# Import libraries, resolve artifact paths, define shared diagnostics helpers, and display the persisted-artifact inventory.

# %%
# ruff: noqa: E501
import json
from pathlib import Path

import pandas as pd
from IPython.display import display

PROJECT_ROOT = next(
    (
        parent
        for parent in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
        if (parent / "pyproject.toml").is_file()
    ),
    Path.cwd().resolve(),
)
RESULTS_DIR = PROJECT_ROOT / "results"

ARTIFACT_DEFAULTS = {
    "prediction_log": "prediction_log.csv",
    "artifact_manifest": "artifact_manifest.json",
    "probe_predictions": "probe_predictions.csv",
    "stress_probe_predictions": "stress_probe_predictions.csv",
    "shortcut_probe_predictions": "shortcut_probe_predictions.csv",
    "stored_attributions": "stored_attributions.csv",
}
ARTIFACT_PARAMETERS = {
    "prediction_log": prediction_log_path,
    "artifact_manifest": artifact_manifest_path,
    "probe_predictions": probe_predictions_path,
    "stress_probe_predictions": stress_probe_predictions_path,
    "shortcut_probe_predictions": shortcut_probe_predictions_path,
    "stored_attributions": stored_attributions_path,
}


def resolve_artifact_path(name: str) -> Path:
    configured_path = ARTIFACT_PARAMETERS[name]
    return (
        Path(configured_path)
        if configured_path is not None
        else RESULTS_DIR / ARTIFACT_DEFAULTS[name]
    )


ARTIFACT_PATHS = {name: resolve_artifact_path(name) for name in ARTIFACT_DEFAULTS}


def needs_new_evidence(section: str, requirement: str, detail: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "section": section,
                "evidence_status": "needs-new-evidence",
                "requirement": requirement,
                "detail": detail,
            }
        ]
    )


def read_csv_or_diagnostic(name: str) -> tuple[pd.DataFrame | None, str | None]:
    path = ARTIFACT_PATHS[name]
    if not path.is_file():
        return None, f"needs-new-evidence: missing {name}: {path}"
    try:
        return pd.read_csv(path), None
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        return None, f"needs-new-evidence: unreadable {name}: {path}: {error}"


def read_manifest_or_diagnostic() -> tuple[dict[str, object] | None, str | None]:
    path = ARTIFACT_PATHS["artifact_manifest"]
    if not path.is_file():
        return None, f"needs-new-evidence: missing artifact_manifest: {path}"
    try:
        loaded_manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"needs-new-evidence: invalid artifact_manifest: {path}: {error}"
    if not isinstance(loaded_manifest, dict):
        return (
            None,
            f"needs-new-evidence: artifact_manifest must be a JSON object: {path}",
        )
    return loaded_manifest, None


manifest, manifest_diagnostic = read_manifest_or_diagnostic()
artifact_inventory = pd.DataFrame(
    [
        {
            "artifact": name,
            "path": str(path.relative_to(PROJECT_ROOT))
            if path.is_relative_to(PROJECT_ROOT)
            else str(path),
            "exists": path.is_file(),
            "status": "available" if path.is_file() else "needs-new-evidence",
            "release_required": True,
        }
        for name, path in ARTIFACT_PATHS.items()
    ]
)
display(artifact_inventory)

# %% [markdown]
# ## 0. Audit Contract and Frozen Candidate
#
# Establish whether persisted artifacts identify one candidate and its evaluation split.

# %%
identity_columns = ("candidate_id", "tokenizer_id", "checkpoint_id")
prediction_log_raw, prediction_log_diagnostic = read_csv_or_diagnostic("prediction_log")
if prediction_log_raw is None:
    contract_evidence = needs_new_evidence(
        "audit_contract",
        "labeled prediction_log",
        prediction_log_diagnostic or "needs-new-evidence",
    )
else:
    present_identity_columns = [
        column for column in identity_columns if column in prediction_log_raw.columns
    ]
    contract_evidence = pd.DataFrame(
        [
            {
                "evidence_status": "available"
                if len(present_identity_columns) == len(identity_columns)
                else "needs-new-evidence",
                "rows": len(prediction_log_raw),
                "identity_columns_present": len(present_identity_columns),
                "manifest_status": "available"
                if manifest_diagnostic is None
                else manifest_diagnostic,
            }
        ]
    )
display(contract_evidence)

# %% [markdown]
# ### Audit Contract Decision
#
# **Observation:** The preceding contract evidence reports whether the persisted prediction log and candidate identifiers are available.
#
# **Interpretation:** A frozen-candidate claim is supported only when the displayed identity fields and manifest status are available.
#
# **Action:** Use the displayed missing requirement to persist or bind the evidence before treating later metrics as a final audit.

# %% [markdown]
# ## 1. Load Prediction, Split, and Model Artifacts
#
# Validate only an existing prediction log; no model artifact is loaded for inference.


# %%
def validate_prediction_log(
    prediction_log: pd.DataFrame,
    candidate_id: str | None,
    tokenizer_id: str | None,
    checkpoint_id: str | None,
    evaluation_split: str = "validation",
) -> pd.DataFrame:
    audit_identity = {
        "candidate_id": candidate_id,
        "tokenizer_id": tokenizer_id,
        "checkpoint_id": checkpoint_id,
    }
    required_columns = {
        "id",
        "split",
        "y_true",
        "y_pred",
        "search_term",
        "product_title",
        "product_description",
    } | set(audit_identity)
    missing_columns = required_columns - set(prediction_log.columns)
    if missing_columns:
        raise ValueError(
            f"Prediction log is missing columns: {sorted(missing_columns)}"
        )
    if not isinstance(evaluation_split, str) or evaluation_split not in {
        "validation",
        "holdout",
        "test",
    }:
        raise ValueError(
            "evaluation_split must be one of: validation, holdout, or test."
        )
    if not bool(prediction_log["id"].notna().all()):
        raise ValueError("id must not contain null values.")
    if bool(prediction_log["id"].duplicated().any()):
        raise ValueError("id values must be unique for final evaluation.")
    for column, value in audit_identity.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{column} parameter must be a non-empty string.")
        if not bool(prediction_log[column].notna().all()) or not bool(
            prediction_log[column].eq(value).all()
        ):
            raise ValueError(
                f"Prediction log {column} must equal supplied parameter {value!r} for every row."
            )
    if not bool(prediction_log["y_true"].notna().all()):
        raise ValueError("y_true is required; refusing an unlabeled Kaggle submission.")
    for column in ("y_true", "y_pred"):
        values = prediction_log[column]
        if not pd.api.types.is_numeric_dtype(values) or not bool(values.notna().all()):
            raise ValueError(f"{column} must be numeric and non-null.")
        if bool(values.isin((float("inf"), float("-inf"))).any()):
            raise ValueError(f"{column} must contain only finite values.")
    if not bool(prediction_log["y_true"].between(1.0, 3.0).all()):
        raise ValueError(
            "y_true values must be in [1.0, 3.0]; refusing an unlabeled Kaggle submission."
        )
    if not bool(prediction_log["split"].notna().all()) or bool(
        prediction_log["split"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("split must not contain blank values.")
    if not bool(prediction_log["split"].eq(evaluation_split).all()):
        raise ValueError(
            f"prediction log rows must all match evaluation_split {evaluation_split!r}."
        )
    for column in ("search_term", "product_title", "product_description"):
        non_null_values = prediction_log[column].dropna()
        if not bool(non_null_values.map(lambda value: isinstance(value, str)).all()):
            raise ValueError(f"{column} must contain only string values.")
    for column in ("search_term", "product_title"):
        if not bool(prediction_log[column].notna().all()) or bool(
            prediction_log[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"{column} must contain non-blank text.")
    prediction_log = prediction_log.copy()
    prediction_log["product_description"] = prediction_log[
        "product_description"
    ].fillna("")
    return prediction_log


prediction_log = None
inferred_identity: dict[str, str | None] = dict.fromkeys(identity_columns)
if prediction_log_raw is None:
    load_evidence = needs_new_evidence(
        "artifact_load",
        "prediction_log",
        prediction_log_diagnostic or "needs-new-evidence",
    )
else:
    inferred_identity = {
        column: candidate_id
        if column == "candidate_id"
        else tokenizer_id
        if column == "tokenizer_id"
        else checkpoint_id
        for column in identity_columns
    }
    for column in identity_columns:
        values = (
            prediction_log_raw[column]
            if column in prediction_log_raw.columns
            else pd.Series(dtype="object")
        )
        if inferred_identity[column] is None and values.nunique(dropna=True) == 1:
            inferred_identity[column] = str(values.dropna().iloc[0])
    try:
        prediction_log = validate_prediction_log(
            prediction_log_raw, evaluation_split=evaluation_split, **inferred_identity
        )
        load_evidence = pd.DataFrame(
            [
                {
                    "evidence_status": "available",
                    "rows": len(prediction_log),
                    "split": evaluation_split,
                    **inferred_identity,
                }
            ]
        )
    except ValueError as error:
        load_evidence = needs_new_evidence(
            "artifact_load",
            "valid labeled prediction_log",
            f"needs-new-evidence: {error}",
        )
display(load_evidence)

# %% [markdown]
# ### Prediction Log Contract Decision
#
# **Observation:** The preceding load evidence records whether the stored log passed its labeled, split, and identity contract.
#
# **Interpretation:** Only a displayed `available` contract permits the downstream metrics to use those persisted rows.
#
# **Action:** Correct the displayed contract gap in the stored artifact instead of generating replacement predictions in this notebook.

# %% [markdown]
# ## 2. Final Performance and Uncertainty
#
# Calculate performance from validated labeled predictions when they are available.

# %%
if prediction_log is None:
    performance_evidence = needs_new_evidence(
        "performance",
        "validated labeled prediction_log",
        "needs-new-evidence: performance metrics require the Section 1 contract",
    )
else:
    residual = prediction_log["y_true"] - prediction_log["y_pred"]
    performance_evidence = pd.DataFrame(
        [
            {
                "evidence_status": "available",
                "rows": len(prediction_log),
                "mae": residual.abs().mean(),
                "rmse": (residual.pow(2).mean()) ** 0.5,
                "mean_residual": residual.mean(),
            }
        ]
    )
display(performance_evidence)

# %% [markdown]
# ### Final Performance Decision
#
# **Observation:** The preceding performance evidence contains real MAE, RMSE, and residual statistics only when the labeled-log contract passed.
#
# **Interpretation:** The displayed row count defines the persisted evaluation evidence behind every reported metric.
#
# **Action:** Use the displayed diagnostic to obtain labels or use the displayed metrics in the final release review.

# %% [markdown]
# ## 3. Robustness Tests
#
# Summarize persisted baseline-versus-probe prediction deltas without creating probe inputs or running inference.

# %%
probe_predictions, probe_diagnostic = read_csv_or_diagnostic("probe_predictions")
if probe_predictions is None:
    robustness_evidence = needs_new_evidence(
        "robustness", "probe_predictions", probe_diagnostic or "needs-new-evidence"
    )
else:
    required_probe_columns = {
        "id",
        "scenario",
        "baseline_prediction",
        "probe_prediction",
    }
    missing_probe_columns = required_probe_columns - set(probe_predictions.columns)
    if missing_probe_columns:
        robustness_evidence = needs_new_evidence(
            "robustness",
            "paired probe predictions",
            f"needs-new-evidence: missing columns {sorted(missing_probe_columns)}",
        )
    else:
        probe_delta = (
            probe_predictions["probe_prediction"]
            - probe_predictions["baseline_prediction"]
        ).abs()
        robustness_evidence = (
            probe_predictions.assign(absolute_prediction_delta=probe_delta)
            .groupby("scenario", dropna=False)
            .agg(
                rows=("id", "size"),
                mean_absolute_prediction_delta=("absolute_prediction_delta", "mean"),
                max_absolute_prediction_delta=("absolute_prediction_delta", "max"),
            )
            .reset_index()
            .assign(evidence_status="available")
        )
display(robustness_evidence)

# %% [markdown]
# ### Robustness Evidence Decision
#
# **Observation:** The preceding robustness table reports persisted paired prediction deltas by scenario, or its precise evidence gap.
#
# **Interpretation:** A scenario is measured only when the displayed table contains stored baseline and probe predictions.
#
# **Action:** Persist the missing paired outputs identified above before making an invariance claim.

# %% [markdown]
# ## 4. Stress Tests
#
# Summarize persisted long-text, empty-description, or other stress predictions without generating them.

# %%
stress_predictions, stress_diagnostic = read_csv_or_diagnostic(
    "stress_probe_predictions"
)
if stress_predictions is None:
    stress_evidence = needs_new_evidence(
        "stress", "stress_probe_predictions", stress_diagnostic or "needs-new-evidence"
    )
else:
    required_stress_columns = {
        "id",
        "scenario",
        "baseline_prediction",
        "probe_prediction",
    }
    missing_stress_columns = required_stress_columns - set(stress_predictions.columns)
    if missing_stress_columns:
        stress_evidence = needs_new_evidence(
            "stress",
            "paired stress predictions",
            f"needs-new-evidence: missing columns {sorted(missing_stress_columns)}",
        )
    else:
        stress_evidence = (
            stress_predictions.assign(
                absolute_prediction_delta=(
                    stress_predictions["probe_prediction"]
                    - stress_predictions["baseline_prediction"]
                ).abs()
            )
            .groupby("scenario", dropna=False)
            .agg(
                rows=("id", "size"),
                mean_absolute_prediction_delta=("absolute_prediction_delta", "mean"),
                max_absolute_prediction_delta=("absolute_prediction_delta", "max"),
            )
            .reset_index()
            .assign(evidence_status="available")
        )
display(stress_evidence)

# %% [markdown]
# ### Stress Evidence Decision
#
# **Observation:** The preceding stress table is populated from stored stress predictions or names the missing persisted evidence.
#
# **Interpretation:** Stress resilience is supported only by the displayed scenario-level prediction deltas.
#
# **Action:** Produce the missing artifact outside this audit, then rerun the notebook to inspect the resulting table.

# %% [markdown]
# ## 5. Adversarial / Shortcut Tests
#
# Summarize persisted shortcut-test prediction deltas without constructing adversarial templates.

# %%
shortcut_predictions, shortcut_diagnostic = read_csv_or_diagnostic(
    "shortcut_probe_predictions"
)
if shortcut_predictions is None:
    shortcut_evidence = needs_new_evidence(
        "shortcut_tests",
        "shortcut_probe_predictions",
        shortcut_diagnostic or "needs-new-evidence",
    )
else:
    required_shortcut_columns = {
        "id",
        "scenario",
        "baseline_prediction",
        "probe_prediction",
    }
    missing_shortcut_columns = required_shortcut_columns - set(
        shortcut_predictions.columns
    )
    if missing_shortcut_columns:
        shortcut_evidence = needs_new_evidence(
            "shortcut_tests",
            "paired shortcut predictions",
            f"needs-new-evidence: missing columns {sorted(missing_shortcut_columns)}",
        )
    else:
        shortcut_evidence = (
            shortcut_predictions.assign(
                absolute_prediction_delta=(
                    shortcut_predictions["probe_prediction"]
                    - shortcut_predictions["baseline_prediction"]
                ).abs()
            )
            .groupby("scenario", dropna=False)
            .agg(
                rows=("id", "size"),
                mean_absolute_prediction_delta=("absolute_prediction_delta", "mean"),
                max_absolute_prediction_delta=("absolute_prediction_delta", "max"),
            )
            .reset_index()
            .assign(evidence_status="available")
        )
display(shortcut_evidence)

# %% [markdown]
# ### Shortcut Test Decision
#
# **Observation:** The preceding shortcut table uses only stored paired outcomes and otherwise renders the missing-evidence diagnostic.
#
# **Interpretation:** Shortcut behavior cannot be inferred from a probe template, submission file, or model weight.
#
# **Action:** Persist the scenario outputs named by the diagnostic before accepting or rejecting shortcut-risk hypotheses.

# %% [markdown]
# ## 6. Error Slices and Failure Taxonomy
#
# Calculate residual slices and a data-derived failure taxonomy from validated prediction rows.

# %%
if prediction_log is None:
    error_slice_evidence = needs_new_evidence(
        "error_slices",
        "validated labeled prediction_log",
        "needs-new-evidence: residual slices require the Section 1 contract",
    )
    failure_taxonomy_summary = None
else:
    residual_frame = prediction_log.assign(
        residual=prediction_log["y_true"] - prediction_log["y_pred"]
    )
    residual_frame["absolute_error"] = residual_frame["residual"].abs()
    residual_frame["description_presence"] = residual_frame["product_description"].ne(
        ""
    )
    residual_frame["query_length_bin"] = pd.cut(
        residual_frame["search_term"].str.len(), [0, 5, 15, 30, float("inf")]
    )
    residual_frame["relevance_band"] = pd.cut(
        residual_frame["y_true"], [0, 1.5, 2.5, float("inf")]
    )
    error_slice_evidence = pd.concat(
        [
            residual_frame.groupby(dimension, dropna=False)
            .agg(
                rows=("id", "size"),
                mae=("absolute_error", "mean"),
                rmse=("residual", lambda values: (values.pow(2).mean()) ** 0.5),
            )
            .reset_index()
            .assign(slice_dimension=dimension, evidence_status="available")
            for dimension in (
                "description_presence",
                "query_length_bin",
                "relevance_band",
                "split",
            )
        ],
        ignore_index=True,
    )
    failure_taxonomy_summary = (
        residual_frame.assign(
            failure_taxonomy=lambda frame: pd.Series("other", index=frame.index).mask(
                frame["absolute_error"].ge(frame["absolute_error"].quantile(0.9)),
                "high_absolute_error",
            )
        )
        .groupby("failure_taxonomy", dropna=False)
        .agg(rows=("id", "size"), mean_absolute_error=("absolute_error", "mean"))
        .reset_index()
    )
display(error_slice_evidence)
if prediction_log is not None:
    display(failure_taxonomy_summary)

# %% [markdown]
# ### Error Slice Decision
#
# **Observation:** The preceding slice table derives errors from the same validated labeled rows used for final performance.
#
# **Interpretation:** The displayed slice dimensions identify where persisted residual evidence is strongest or absent.
#
# **Action:** Investigate the largest displayed slice error, or satisfy the diagnostic before creating a failure taxonomy claim.

# %% [markdown]
# ## 7. XAI and Explanation Validation
#
# Summarize stored attribution faithfulness only when linked to validated prediction IDs.

# %%
stored_attributions, attribution_diagnostic = read_csv_or_diagnostic(
    "stored_attributions"
)
if prediction_log is None:
    xai_evidence = needs_new_evidence(
        "xai",
        "validated prediction_log",
        "needs-new-evidence: attribution linkage requires the Section 1 contract",
    )
elif stored_attributions is None:
    xai_evidence = needs_new_evidence(
        "xai", "stored_attributions", attribution_diagnostic or "needs-new-evidence"
    )
else:
    required_attribution_columns = {
        "id",
        "faithfulness_deletion_delta",
        *identity_columns,
    }
    missing_attribution_columns = required_attribution_columns - set(
        stored_attributions.columns
    )
    if missing_attribution_columns:
        xai_evidence = needs_new_evidence(
            "xai",
            "linked attribution columns",
            f"needs-new-evidence: missing columns {sorted(missing_attribution_columns)}",
        )
    elif not bool(stored_attributions["id"].isin(prediction_log["id"]).all()):
        xai_evidence = needs_new_evidence(
            "xai",
            "linked attribution IDs",
            "needs-new-evidence: stored attribution IDs are not all present in prediction_log",
        )
    elif any(
        not bool(stored_attributions[column].eq(inferred_identity[column]).all())
        for column in identity_columns
    ):
        xai_evidence = needs_new_evidence(
            "xai",
            "linked attribution identity",
            "needs-new-evidence: stored attribution identity does not match prediction_log",
        )
    else:
        faithfulness = stored_attributions["faithfulness_deletion_delta"]
        if (
            not pd.api.types.is_numeric_dtype(faithfulness)
            or not bool(faithfulness.notna().all())
            or bool(faithfulness.isin((float("inf"), float("-inf"))).any())
        ):
            xai_evidence = needs_new_evidence(
                "xai",
                "numeric faithfulness_deletion_delta",
                "needs-new-evidence: attribution faithfulness values must be numeric and non-null",
            )
        else:
            xai_evidence = pd.DataFrame(
                [
                    {
                        "evidence_status": "available",
                        "attribution_rows": len(stored_attributions),
                        "mean_faithfulness_deletion_delta": faithfulness.mean(),
                        "positive_faithfulness_rate": faithfulness.gt(0).mean(),
                    }
                ]
            )
display(xai_evidence)

# %% [markdown]
# ### Explanation Validity Decision
#
# **Observation:** The preceding XAI evidence reports stored, linked faithfulness values or the exact linkage or quality gap.
#
# **Interpretation:** Explanation validity depends on the displayed attribution rows and their linkage to evaluated predictions.
#
# **Action:** Store or repair the attribution evidence named above before using explanations in a release decision.

# %% [markdown]
# ## 8. Release Decision and Known Limitations
#
# Derive the decision only from the displayed artifact-completeness status table.

# %%
release_artifact_status = artifact_inventory.assign(
    complete=artifact_inventory["status"].eq("available")
)
release_decision = (
    "submit"
    if bool(release_artifact_status["complete"].all())
    else "needs-new-evidence"
)
release_evidence = release_artifact_status.assign(release_decision=release_decision)
display(release_evidence)

# %% [markdown]
# ### Release Decision Record
#
# **Observation:** The preceding release table derives its decision from the artifact-completeness status of every required persisted input.
#
# **Interpretation:** An incomplete displayed inventory prevents a release decision based on evidence that is not yet stored.
#
# **Action:** Resolve each displayed `needs-new-evidence` row, then rerun this audit to derive the next decision from the refreshed inventory.
