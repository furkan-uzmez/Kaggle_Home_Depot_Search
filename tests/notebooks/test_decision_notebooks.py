import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import nbformat
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


NOTEBOOK_HEADINGS = {
    "01_data_contract_and_eda.ipynb": (
        "## 0. Problem Contract",
        "## 1. Reproducible Setup",
        "## 2. Raw Data and Join Contract",
        "## 3. Target and Text-Field Audit",
        "## 4. Text Quality and Length Profiling",
        "## 5. Duplicate, Near-Duplicate, and Leakage Audit",
        "## 6. Query/Product Dependency and CV Decision",
        "## 7. TF-IDF and DeBERTa Tokenizer Readiness",
        "## 8. Evidence-Based Preprocessing Contract",
        "## 9. Findings and Next Actions",
    ),
    "02_classical_models_and_hpo_review.ipynb": (
        "## 0. Model-Selection Contract",
        "## 1. Load Frozen Experiment Artifacts",
        "## 2. Validation and Pipeline Boundary Audit",
        "## 3. Baseline vs HPO Ridge Comparison",
        "## 4. Classical vs DeBERTa Comparison",
        "## 5. Regression Residual and Slice Analysis",
        "## 6. Failure Review",
        "## 7. Submission Sanity Checks",
        "## 8. Final Candidate Decision",
    ),
    "03_final_model_audit.ipynb": (
        "## 0. Audit Contract and Frozen Candidate",
        "## 1. Load Prediction, Split, and Model Artifacts",
        "## 2. Final Performance and Uncertainty",
        "## 3. Robustness Tests",
        "## 4. Stress Tests",
        "## 5. Adversarial / Shortcut Tests",
        "## 6. Error Slices and Failure Taxonomy",
        "## 7. XAI and Explanation Validation",
        "## 8. Release Decision and Known Limitations",
    ),
}


def read_notebook(filename: str) -> nbformat.NotebookNode:
    notebook_path = NOTEBOOKS_DIR / filename
    assert notebook_path.is_file(), f"Missing decision notebook: {notebook_path}"
    return nbformat.read(notebook_path, as_version=4)


def code_source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")


def notebook_source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(cell.source for cell in notebook.cells)


def prediction_log_validator() -> Callable[..., pd.DataFrame]:
    source = code_source(read_notebook("03_final_model_audit.ipynb"))
    function_match = re.search(
        r"def validate_prediction_log[\s\S]*?(?=\n\nprediction_log\s*=|\n# %%)",
        source,
    )
    assert function_match is not None
    namespace = {"pd": pd}
    exec(function_match.group(), namespace)
    return cast(Callable[..., pd.DataFrame], namespace["validate_prediction_log"])


def valid_audit_identity() -> dict[str, str]:
    return {
        "candidate_id": "deberta-v1",
        "tokenizer_id": "deberta-tokenizer-v1",
        "checkpoint_id": "checkpoint-1000",
    }


def valid_prediction_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1],
            "split": ["validation"],
            "y_true": [2.0],
            "y_pred": [2.1],
            "search_term": ["hammer"],
            "product_title": ["Steel Hammer"],
            "product_description": ["A steel hammer"],
            **valid_audit_identity(),
        }
    )


def run_submission_validation(
    submission_path: Path | None,
    *,
    selected_experiment: str = "ridge-v1",
    submission_artifact_name: str | None = None,
    artifact_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    notebook = read_notebook("02_classical_models_and_hpo_review.ipynb")
    artifact_selection_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "artifact_directories = " in cell.source
    )
    submission_marker = 'submission_path = optional_artifacts.get("submission")'
    submission_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and submission_marker in cell.source
    )
    diagnostics: list[str] = []
    namespace: dict[str, object] = {
        "Path": Path,
        "pd": pd,
        "submission_artifact_name": submission_artifact_name,
        "selected_experiment": selected_experiment,
        "results_dir": (
            submission_path.parent if submission_path is not None else Path.cwd()
        ),
        "PROJECT_ROOT": (
            submission_path.parent.parent if submission_path is not None else Path.cwd()
        ),
        "oof_artifact_name": None,
        "fold_artifact_name": None,
        "artifact_diagnostics": diagnostics,
        "needs_new_evidence": lambda message: diagnostics.append(
            f"needs-new-evidence: {message}"
        ),
    }
    exec(artifact_selection_cell, namespace)
    namespace["artifact_manifest"] = artifact_manifest or {}
    exec(submission_cell, namespace)
    return namespace


def test_data_contract_notebook_uses_configured_loader_contract() -> None:
    source = code_source(read_notebook("01_data_contract_and_eda.ipynb"))

    assert 'data_config = yaml.safe_load(handle)["data"]' in source
    assert 'data_dir = PROJECT_ROOT / Path(data_config["train_path"]).parent' in source
    assert 'encoding="ISO-8859-1"' in source
    assert "dtype=" in source


def test_model_review_notebook_uses_rmse_when_available() -> None:
    source = code_source(read_notebook("02_classical_models_and_hpo_review.ipynb"))

    assert 'metric_column = "rmse"' in source


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("id", None, "id"),
        ("split", None, "split"),
        ("split", "   ", "split"),
        ("y_true", None, "unlabeled Kaggle submission"),
        ("y_true", "not-a-number", "y_true"),
        ("y_true", float("inf"), "y_true"),
        ("y_pred", None, "y_pred"),
        ("y_pred", "not-a-number", "y_pred"),
        ("y_pred", float("inf"), "y_pred"),
    ),
)
def test_prediction_log_validator_rejects_invalid_values(
    column: str, value: object, message: str
) -> None:
    prediction_log = valid_prediction_log()
    if isinstance(value, str):
        prediction_log[column] = pd.Series([value], dtype="object")
    else:
        prediction_log.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        prediction_log_validator()(prediction_log, **valid_audit_identity())


def test_prediction_log_validator_rejects_non_string_search_term() -> None:
    prediction_log = valid_prediction_log()
    prediction_log["search_term"] = pd.Series([123], dtype="object")

    with pytest.raises(ValueError, match="search_term"):
        prediction_log_validator()(prediction_log, **valid_audit_identity())


def test_prediction_log_validator_normalizes_nullable_description() -> None:
    prediction_log = valid_prediction_log()
    prediction_log.loc[0, "product_description"] = None

    validated = prediction_log_validator()(prediction_log, **valid_audit_identity())

    assert validated.loc[0, "product_description"] == ""


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("id", 1, "unique"),
        ("split", "test", "evaluation_split"),
        ("y_true", 0.9, "[1.0, 3.0]"),
        ("y_true", 3.1, "[1.0, 3.0]"),
    ),
)
def test_prediction_log_validator_rejects_invalid_evaluation_contract(
    column: str, value: object, message: str
) -> None:
    prediction_log = pd.concat([valid_prediction_log(), valid_prediction_log()])
    prediction_log = prediction_log.reset_index(drop=True)
    prediction_log.loc[1, "id"] = 2
    prediction_log.loc[1, column] = value

    with pytest.raises(ValueError, match=re.escape(message)):
        prediction_log_validator()(prediction_log, **valid_audit_identity())


def test_prediction_log_validator_accepts_configured_evaluation_split() -> None:
    prediction_log = valid_prediction_log()
    prediction_log["split"] = "holdout"

    validated = prediction_log_validator()(
        prediction_log,
        **valid_audit_identity(),
        evaluation_split="holdout",
    )

    assert validated["split"].tolist() == ["holdout"]


@pytest.mark.parametrize("parameter", ("candidate_id", "tokenizer_id", "checkpoint_id"))
@pytest.mark.parametrize("value", (None, "", "   "))
def test_prediction_log_validator_requires_non_empty_audit_identity(
    parameter: str, value: object
) -> None:
    identity: dict[str, object] = dict(valid_audit_identity())
    identity[parameter] = value

    with pytest.raises(ValueError, match=parameter):
        prediction_log_validator()(valid_prediction_log(), **identity)


@pytest.mark.parametrize("column", ("candidate_id", "tokenizer_id", "checkpoint_id"))
def test_prediction_log_validator_rejects_mismatched_identity(column: str) -> None:
    prediction_log = valid_prediction_log()
    prediction_log.loc[0, column] = "other-identity"

    with pytest.raises(ValueError, match=column):
        prediction_log_validator()(prediction_log, **valid_audit_identity())


def test_prediction_log_validator_rejects_missing_identity_column() -> None:
    prediction_log = valid_prediction_log().drop(columns="checkpoint_id")

    with pytest.raises(ValueError, match="checkpoint_id"):
        prediction_log_validator()(prediction_log, **valid_audit_identity())


def test_notebook_two_and_three_include_review_artifact_contracts() -> None:
    model_source = code_source(
        read_notebook("02_classical_models_and_hpo_review.ipynb")
    )
    audit_source = code_source(read_notebook("03_final_model_audit.ipynb"))

    assert 'discover_artifacts("*_oof.csv")' in model_source
    assert 'discover_artifacts("*_folds.csv")' in model_source
    assert (
        'rename(columns={"relevance": "y_true", "prediction": "y_pred"})'
        in model_source
    )
    for name in (
        "query_length_bin",
        "description_presence",
        "relevance_band",
        "failure_taxonomy",
    ):
        assert name in audit_source
    for name in (
        "candidate_id",
        "tokenizer_id",
        "checkpoint_id",
    ):
        assert name in audit_source
    assert "faithfulness" in audit_source


def test_notebooks_include_taxonomy_attribution_and_manifest_selection() -> None:
    model_source = code_source(
        read_notebook("02_classical_models_and_hpo_review.ipynb")
    )
    audit_source = code_source(read_notebook("03_final_model_audit.ipynb"))

    assert "artifact_manifest_path = None" in model_source
    assert "failure_taxonomy_summary" in audit_source
    assert "faithfulness_deletion_delta" in audit_source
    assert "positive_faithfulness_rate" in audit_source


def test_candidate_binding_and_attribution_linkage_contracts() -> None:
    model_source = code_source(
        read_notebook("02_classical_models_and_hpo_review.ipynb")
    )
    audit_source = code_source(read_notebook("03_final_model_audit.ipynb"))

    for name in (
        "selected_experiment",
        "artifact_manifest_path",
        "oof_artifact_name",
        "fold_artifact_name",
    ):
        assert name in model_source
    assert "oof_candidate_name" not in model_source
    assert "fold_candidate_name" not in model_source
    assert "needs-new-evidence" in model_source
    assert 'optional_artifacts.get("submission")' in model_source
    assert "linked attribution IDs" in audit_source
    assert '.isin(prediction_log["id"]).all()' in audit_source


def test_model_review_discovers_project_artifact_locations() -> None:
    model_source = code_source(
        read_notebook("02_classical_models_and_hpo_review.ipynb")
    )

    for path in (
        'results_dir / "experiment_summary.csv"',
        'PROJECT_ROOT / "logs" / "experiments.csv"',
        'results_dir / "hpo_results.json"',
        'PROJECT_ROOT / "logs" / "hpo" / "hpo_results.json"',
        'PROJECT_ROOT / "artifacts"',
    ):
        assert path in model_source


def test_submission_validation_reports_missing_selected_path_without_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_submission_path = tmp_path / "submission.csv"

    def fail_if_read(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("read_csv must not run for a missing submission path")

    monkeypatch.setattr(pd, "read_csv", fail_if_read)

    namespace = run_submission_validation(
        missing_submission_path,
        submission_artifact_name="submission.csv",
        artifact_manifest={
            "submission.csv": {"experiment": "ridge-v1", "kind": "submission"}
        },
    )

    unavailable_prefix = "needs-new-evidence: selected submission artifact is "
    unavailable_prefix += "not among discovered candidates: "
    expected_diagnostic = unavailable_prefix + "submission.csv"
    assert namespace["artifact_diagnostics"] == [expected_diagnostic]
    assert "submission_checks" not in namespace


def test_submission_validation_accepts_exact_finite_submission_schema(
    tmp_path: Path,
) -> None:
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "relevance": [1.0, 3.0]}).to_csv(
        submission_path, index=False
    )

    namespace = run_submission_validation(
        submission_path,
        submission_artifact_name="submission.csv",
        artifact_manifest={
            "submission.csv": {"experiment": "ridge-v1", "kind": "submission"}
        },
    )

    checks = cast(pd.DataFrame, namespace["submission_checks"])
    assert checks.loc[0, "columns"] == ["id", "relevance"]
    assert checks.loc[0, "relevance_min"] == 1.0
    assert checks.loc[0, "relevance_max"] == 3.0


@pytest.mark.parametrize(
    ("submission", "message"),
    (
        (pd.DataFrame({"id": [1], "prediction": [2.0]}), "exactly"),
        (pd.DataFrame({"id": [None], "relevance": [2.0]}), "id"),
        (pd.DataFrame({"id": [1, 1], "relevance": [2.0, 2.5]}), "unique"),
        (pd.DataFrame({"id": ["one"], "relevance": [2.0]}), "id"),
        (pd.DataFrame({"id": [1], "relevance": [float("inf")]}), "relevance"),
        (pd.DataFrame({"id": [1], "relevance": [0.9]}), "[1.0, 3.0]"),
    ),
)
def test_submission_validation_rejects_malformed_submission_data(
    tmp_path: Path, submission: pd.DataFrame, message: str
) -> None:
    submission_path = tmp_path / "submission.csv"
    submission.to_csv(submission_path, index=False)

    with pytest.raises(ValueError, match=re.escape(message)):
        run_submission_validation(
            submission_path,
            submission_artifact_name="submission.csv",
            artifact_manifest={
                "submission.csv": {"experiment": "ridge-v1", "kind": "submission"}
            },
        )


def run_model_review_artifact_analysis(
    results_dir: Path,
    *,
    selected_experiment: str,
    oof_artifact_name: str | None = None,
    fold_artifact_name: str | None = None,
    submission_artifact_name: str | None = None,
    artifact_manifest_path: Path | None = None,
) -> dict[str, object]:
    notebook = read_notebook("02_classical_models_and_hpo_review.ipynb")
    artifact_selection_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "oof_candidates = " in cell.source
    )
    artifact_analysis_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "oof_path = optional_artifacts" in cell.source
    )
    namespace: dict[str, object] = {
        "Path": Path,
        "json": json,
        "pd": pd,
        "results_dir": results_dir,
        "PROJECT_ROOT": results_dir.parent,
        "selected_experiment": selected_experiment,
        "oof_artifact_name": oof_artifact_name,
        "fold_artifact_name": fold_artifact_name,
        "submission_artifact_name": submission_artifact_name,
        "artifact_manifest_path": artifact_manifest_path,
    }
    exec(artifact_selection_cell, namespace)
    exec(artifact_analysis_cell, namespace)
    return namespace


def run_join_contract_validation(
    train: pd.DataFrame,
    product_descriptions: pd.DataFrame,
    attributes: pd.DataFrame,
) -> dict[str, object]:
    notebook = read_notebook("01_data_contract_and_eda.ipynb")
    join_contract_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "joined_train = train.merge" in cell.source
    )
    namespace: dict[str, object] = {
        "pd": pd,
        "train": train,
        "product_descriptions": product_descriptions,
        "attributes": attributes,
    }
    exec(join_contract_cell, namespace)
    return namespace


def valid_join_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            {
                "product_uid": [1, 1, 2],
                "id": [10, 11, 12],
                "search_term": ["one", "one", "two"],
                "product_title": ["One", "One", "Two"],
                "relevance": [2.0, 2.0, 2.0],
            }
        ),
        pd.DataFrame({"product_uid": [1, 2], "product_description": ["one", "two"]}),
        pd.DataFrame(
            {
                "product_uid": [1, 1, 2],
                "name": ["brand", "color", "brand"],
                "value": ["Acme", "red", "Bolt"],
            }
        ),
    )


def test_data_contract_join_audit_records_join_key_and_row_preservation_evidence() -> (
    None
):
    namespace = run_join_contract_validation(*valid_join_tables())

    join_contract = cast(pd.DataFrame, namespace["join_contract"])
    assert bool(join_contract["key_unique"].all())
    assert join_contract["rows_before"].tolist() == [3, 3]
    assert join_contract["rows_after"].tolist() == [3, 3]


@pytest.mark.parametrize(
    ("table_index", "column", "value", "message"),
    (
        (0, "product_uid", None, "train.product_uid"),
        (1, "product_uid", 1, "product_descriptions.product_uid"),
    ),
)
def test_data_contract_join_audit_rejects_invalid_join_keys(
    table_index: int, column: str, value: object, message: str
) -> None:
    tables = list(valid_join_tables())
    tables[table_index].loc[1, column] = value

    with pytest.raises(ValueError, match=re.escape(message)):
        run_join_contract_validation(
            *cast(tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tables)
        )


def write_artifact_manifest(path: Path, artifacts: dict[str, dict[str, str]]) -> Path:
    path.write_text(json.dumps(artifacts), encoding="utf-8")
    return path


def test_model_review_analyzes_only_manifest_bound_selected_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    oof_name = "ridge_oof.csv"
    fold_name = "ridge_folds.csv"
    pd.DataFrame({"relevance": [2.0], "prediction": [2.1]}).to_csv(
        tmp_path / oof_name, index=False
    )
    pd.DataFrame({"fold": [0], "rmse": [0.5]}).to_csv(tmp_path / fold_name, index=False)
    manifest_path = write_artifact_manifest(
        tmp_path / "artifact_manifest.json",
        {
            oof_name: {"experiment": "ridge-v1", "kind": "oof"},
            fold_name: {"experiment": "ridge-v1", "kind": "folds"},
        },
    )

    namespace = run_model_review_artifact_analysis(
        tmp_path,
        selected_experiment="ridge-v1",
        oof_artifact_name=oof_name,
        fold_artifact_name=fold_name,
        artifact_manifest_path=manifest_path,
    )

    oof = cast(pd.DataFrame, namespace["oof"])
    assert oof.loc[0, "residual"] == pytest.approx(-0.1)
    assert "needs-new-evidence" not in capsys.readouterr().out


def test_model_review_skips_when_oof_selection_is_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    oof_name = "ridge_oof.csv"
    pd.DataFrame({"relevance": [2.0], "prediction": [2.1]}).to_csv(
        tmp_path / oof_name, index=False
    )
    manifest_path = write_artifact_manifest(
        tmp_path / "artifact_manifest.json",
        {oof_name: {"experiment": "ridge-v1", "kind": "oof"}},
    )

    namespace = run_model_review_artifact_analysis(
        tmp_path,
        selected_experiment="ridge-v1",
        artifact_manifest_path=manifest_path,
    )

    assert "oof" not in namespace
    assert namespace["artifact_diagnostics"]


def test_model_review_skips_manifest_without_selected_artifact_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    oof_name = "ridge_oof.csv"
    pd.DataFrame({"relevance": [2.0], "prediction": [2.1]}).to_csv(
        tmp_path / oof_name, index=False
    )
    manifest_path = write_artifact_manifest(
        tmp_path / "artifact_manifest.json",
        {"other_oof.csv": {"experiment": "ridge-v1", "kind": "oof"}},
    )

    namespace = run_model_review_artifact_analysis(
        tmp_path,
        selected_experiment="ridge-v1",
        oof_artifact_name=oof_name,
        artifact_manifest_path=manifest_path,
    )

    assert "oof" not in namespace
    assert namespace["artifact_diagnostics"]


def test_model_review_rejects_fold_manifest_mismatch_without_skipping_valid_oof(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    oof_name = "ridge_oof.csv"
    fold_name = "ridge_folds.csv"
    pd.DataFrame({"relevance": [2.0], "prediction": [2.1]}).to_csv(
        tmp_path / oof_name, index=False
    )
    pd.DataFrame({"fold": [0], "rmse": [0.5]}).to_csv(tmp_path / fold_name, index=False)
    manifest_path = write_artifact_manifest(
        tmp_path / "artifact_manifest.json",
        {
            oof_name: {"experiment": "ridge-v1", "kind": "oof"},
            fold_name: {"experiment": "ridge-v1", "kind": "oof"},
        },
    )

    namespace = run_model_review_artifact_analysis(
        tmp_path,
        selected_experiment="ridge-v1",
        oof_artifact_name=oof_name,
        fold_artifact_name=fold_name,
        artifact_manifest_path=manifest_path,
    )

    oof = cast(pd.DataFrame, namespace["oof"])
    assert oof.loc[0, "residual"] == pytest.approx(-0.1)
    assert "needs-new-evidence" in capsys.readouterr().out


@pytest.mark.parametrize(
    "manifest_entries",
    (
        None,
        {"ridge_oof.csv": {"experiment": "other-v1", "kind": "oof"}},
        {"ridge_oof.csv": {"experiment": "ridge-v1", "kind": "folds"}},
    ),
)
def test_model_review_skips_missing_or_invalid_manifest_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    manifest_entries: dict[str, dict[str, str]] | None,
) -> None:
    oof_name = "ridge_oof.csv"
    pd.DataFrame({"relevance": [2.0], "prediction": [2.1]}).to_csv(
        tmp_path / oof_name, index=False
    )
    manifest_path = (
        None
        if manifest_entries is None
        else write_artifact_manifest(
            tmp_path / "artifact_manifest.json", manifest_entries
        )
    )

    namespace = run_model_review_artifact_analysis(
        tmp_path,
        selected_experiment="ridge-v1",
        oof_artifact_name=oof_name,
        artifact_manifest_path=manifest_path,
    )

    assert "oof" not in namespace
    assert "needs-new-evidence" in capsys.readouterr().out


def test_model_review_skips_malformed_json_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    oof_name = "ridge_oof.csv"
    pd.DataFrame({"relevance": [2.0], "prediction": [2.1]}).to_csv(
        tmp_path / oof_name, index=False
    )
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text("{not valid json", encoding="utf-8")

    namespace = run_model_review_artifact_analysis(
        tmp_path,
        selected_experiment="ridge-v1",
        oof_artifact_name=oof_name,
        artifact_manifest_path=manifest_path,
    )

    assert "oof" not in namespace
    assert "needs-new-evidence" in capsys.readouterr().out


def test_model_review_skips_manifest_when_read_raises_os_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    oof_name = "ridge_oof.csv"
    pd.DataFrame({"relevance": [2.0], "prediction": [2.1]}).to_csv(
        tmp_path / oof_name, index=False
    )
    manifest_path = write_artifact_manifest(
        tmp_path / "artifact_manifest.json",
        {oof_name: {"experiment": "ridge-v1", "kind": "oof"}},
    )
    original_read_text = Path.read_text

    def raise_manifest_read_error(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if path == manifest_path:
            raise OSError("manifest read failed")
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", raise_manifest_read_error)

    namespace = run_model_review_artifact_analysis(
        tmp_path,
        selected_experiment="ridge-v1",
        oof_artifact_name=oof_name,
        artifact_manifest_path=manifest_path,
    )

    assert "oof" not in namespace
    assert "needs-new-evidence" in capsys.readouterr().out


def test_model_review_skips_nonexistent_manifest_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    oof_name = "ridge_oof.csv"
    pd.DataFrame({"relevance": [2.0], "prediction": [2.1]}).to_csv(
        tmp_path / oof_name, index=False
    )

    namespace = run_model_review_artifact_analysis(
        tmp_path,
        selected_experiment="ridge-v1",
        oof_artifact_name=oof_name,
        artifact_manifest_path=tmp_path / "missing_manifest.json",
    )

    assert "oof" not in namespace
    assert "needs-new-evidence" in capsys.readouterr().out


@pytest.mark.parametrize("manifest_contents", ("[]", '"not-an-object"'))
def test_model_review_skips_non_object_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], manifest_contents: str
) -> None:
    oof_name = "ridge_oof.csv"
    pd.DataFrame({"relevance": [2.0], "prediction": [2.1]}).to_csv(
        tmp_path / oof_name, index=False
    )
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(manifest_contents, encoding="utf-8")

    namespace = run_model_review_artifact_analysis(
        tmp_path,
        selected_experiment="ridge-v1",
        oof_artifact_name=oof_name,
        artifact_manifest_path=manifest_path,
    )

    assert "oof" not in namespace
    assert "needs-new-evidence" in capsys.readouterr().out


def test_model_review_skips_manifest_entry_with_non_object_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    oof_name = "ridge_oof.csv"
    pd.DataFrame({"relevance": [2.0], "prediction": [2.1]}).to_csv(
        tmp_path / oof_name, index=False
    )
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps({oof_name: "not-an-object-record"}), encoding="utf-8"
    )

    namespace = run_model_review_artifact_analysis(
        tmp_path,
        selected_experiment="ridge-v1",
        oof_artifact_name=oof_name,
        artifact_manifest_path=manifest_path,
    )

    assert "oof" not in namespace
    assert "needs-new-evidence" in capsys.readouterr().out


def test_model_review_skips_nonexistent_selected_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = write_artifact_manifest(
        tmp_path / "artifact_manifest.json",
        {"missing_oof.csv": {"experiment": "ridge-v1", "kind": "oof"}},
    )

    namespace = run_model_review_artifact_analysis(
        tmp_path,
        selected_experiment="ridge-v1",
        oof_artifact_name="missing_oof.csv",
        artifact_manifest_path=manifest_path,
    )

    assert "oof" not in namespace
    assert "needs-new-evidence" in capsys.readouterr().out


def test_model_review_discovers_artifacts_from_standard_cli_log_directories(
    tmp_path: Path,
) -> None:
    hpo_dir = tmp_path / "logs" / "hpo"
    run_dir = tmp_path / "logs" / "runs"
    hpo_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (hpo_dir / "ridge_oof.csv").write_text("relevance,prediction\n2.0,2.1\n")
    (run_dir / "ridge_folds.csv").write_text("fold,rmse\n0,0.5\n")

    namespace = run_model_review_artifact_analysis(
        tmp_path / "results",
        selected_experiment="ridge-v1",
    )

    oof_candidates = cast(list[Path], namespace["oof_candidates"])
    fold_candidates = cast(list[Path], namespace["fold_candidates"])
    assert [path.name for path in oof_candidates] == ["ridge_oof.csv"]
    assert [path.name for path in fold_candidates] == ["ridge_folds.csv"]


@pytest.mark.parametrize(
    "manifest_record",
    (
        None,
        {"experiment": "other-v1", "kind": "submission"},
        {"experiment": "ridge-v1", "kind": "oof"},
    ),
)
def test_submission_validation_requires_manifest_bound_selected_submission(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    manifest_record: dict[str, str] | None,
) -> None:
    submission_path = tmp_path / "ridge_submission.csv"
    pd.DataFrame({"id": [1], "relevance": [2.0]}).to_csv(submission_path, index=False)
    manifest = (
        {} if manifest_record is None else {"ridge_submission.csv": manifest_record}
    )

    namespace = run_submission_validation(
        submission_path,
        submission_artifact_name="ridge_submission.csv",
        artifact_manifest=manifest,
    )

    assert "submission_checks" not in namespace
    assert namespace["artifact_diagnostics"]


def test_submission_validation_requires_an_explicit_selected_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    submission_path = tmp_path / "ridge_submission.csv"
    pd.DataFrame({"id": [1], "relevance": [2.0]}).to_csv(submission_path, index=False)

    namespace = run_submission_validation(
        submission_path,
        artifact_manifest={
            "ridge_submission.csv": {"experiment": "ridge-v1", "kind": "submission"}
        },
    )

    assert "submission_checks" not in namespace
    assert namespace["artifact_diagnostics"]


@pytest.mark.parametrize(("filename", "headings"), NOTEBOOK_HEADINGS.items())
def test_decision_notebook_has_required_headings(
    filename: str, headings: tuple[str, ...]
) -> None:
    notebook = read_notebook(filename)
    markdown_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    for heading in headings:
        assert heading in markdown_source


@pytest.mark.parametrize("filename", NOTEBOOK_HEADINGS)
def test_decision_notebook_uses_python3_parameters_cell(filename: str) -> None:
    notebook = read_notebook(filename)

    assert notebook.metadata["kernelspec"]["name"] == "python3"
    assert any(
        cell.cell_type == "code" and "parameters" in cell.metadata.get("tags", [])
        for cell in notebook.cells
    )


@pytest.mark.parametrize("filename", NOTEBOOK_HEADINGS)
def test_decision_notebook_code_uses_portable_project_root(filename: str) -> None:
    source = code_source(read_notebook(filename))
    absolute_path_literal = r"""["'](?:/|[A-Za-z]:[\\/]|\\\\)"""

    assert "Path.cwd().resolve()" in source
    assert ".parents" in source
    assert "pyproject.toml" in source
    assert not re.search(rf"Path\s*\(\s*{absolute_path_literal}", source)
    assert not re.search(
        rf"(?:open|(?:[A-Za-z_]\w*\.)?(?:read|write)\w*)\s*\(\s*"
        rf"(?:\w+\s*=\s*)?{absolute_path_literal}",
        source,
    )
    assert not re.search(r"Path\s*\.\s*home\s*\(", source)
    assert not re.search(r"\.expanduser\s*\(", source)


@pytest.mark.parametrize("filename", NOTEBOOK_HEADINGS)
def test_decision_notebook_is_review_only(filename: str) -> None:
    source = code_source(read_notebook(filename))
    forbidden_patterns = (
        r"\.fit\s*\(",
        r"\.fit_generator\s*\(",
        r"\.fit_transform\s*\(",
        r"\.train\s*\(",
        r"study\.optimize\s*\(",
        r"optuna\.create_study\s*\(",
        r"cross_val_score\s*\(",
        r"cross_validate\s*\(",
        r"GridSearchCV\s*\(",
        r"RandomizedSearchCV\s*\(",
        r"\b(?:submission|predictions|preds)(?:_\w+)?\s*\.\s*to_csv\s*\(",
        r"\bcompetition_submit\s*\(",
    )
    for pattern in forbidden_patterns:
        assert not re.search(pattern, source)


def test_final_model_audit_requires_labels_and_submission_limitation() -> None:
    notebook = read_notebook("03_final_model_audit.ipynb")
    source = code_source(notebook)
    parameter_cells = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "parameters" in cell.metadata.get("tags", [])
    ]
    required_columns = (
        "id",
        "split",
        "y_true",
        "y_pred",
        "search_term",
        "product_title",
        "product_description",
    )
    validation_match = re.search(
        r"def\s+validate_prediction_log[\s\S]*?(?=\ndef\s|\Z)", source
    )
    refusal_pattern = (
        r"raise\s+ValueError\([\s\S]{0,500}?(?:"
        r"y_true[\s\S]{0,500}?unlabeled Kaggle submission|"
        r"unlabeled Kaggle submission[\s\S]{0,500}?y_true)"
    )

    assert any("prediction_log_path" in cell for cell in parameter_cells)
    assert "needs-new-evidence" in source
    assert validation_match is not None
    validation_source = validation_match.group()
    assert "required_columns" in validation_source
    assert "prediction_log.columns" in validation_source
    assert "missing_columns" in validation_source
    assert "raise ValueError" in validation_source
    for column in required_columns:
        assert column in validation_source
    assert re.search(refusal_pattern, validation_source)
    validated_prediction_log = source.index("validate_prediction_log(")
    for metric_name in ("mean_squared_error", "mean_absolute_error"):
        if metric_name in source:
            assert validated_prediction_log < source.index(metric_name)
    assert "unlabeled Kaggle submission" in notebook_source(notebook)


def test_final_model_audit_rejects_training_split() -> None:
    validator = prediction_log_validator()

    with pytest.raises(ValueError, match="validation, holdout, or test"):
        validator(
            valid_prediction_log().assign(split="train"),
            **valid_audit_identity(),
            evaluation_split="train",
        )


def test_final_model_audit_execution_with_mock_artifacts(tmp_path: Path) -> None:
    prediction_log = pd.DataFrame({
        "id": [1, 2],
        "split": ["validation", "validation"],
        "y_true": [2.0, 3.0],
        "y_pred": [2.1, 2.9],
        "search_term": ["hammer", "drill"],
        "product_title": ["Steel Hammer", "Power Drill"],
        "product_description": ["A hammer", "A drill"],
        "candidate_id": ["deberta-v1", "deberta-v1"],
        "tokenizer_id": ["deberta-tokenizer-v1", "deberta-tokenizer-v1"],
        "checkpoint_id": ["checkpoint-1000", "checkpoint-1000"]
    })
    prediction_log_path = tmp_path / "prediction_log.csv"
    prediction_log.to_csv(prediction_log_path, index=False)

    manifest_data = {
        "prediction_log.csv": {"experiment": "deberta-v1", "kind": "prediction_log"},
        "artifact_manifest.json": {"experiment": "deberta-v1", "kind": "manifest"}
    }
    manifest_path = tmp_path / "artifact_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    probe_predictions = pd.DataFrame({
        "id": [1, 2],
        "scenario": ["synonym", "synonym"],
        "baseline_prediction": [2.1, 2.9],
        "probe_prediction": [2.2, 2.8]
    })
    probe_path = tmp_path / "probe_predictions.csv"
    stress_path = tmp_path / "stress_probe_predictions.csv"
    shortcut_path = tmp_path / "shortcut_probe_predictions.csv"
    probe_predictions.to_csv(probe_path, index=False)
    probe_predictions.to_csv(stress_path, index=False)
    probe_predictions.to_csv(shortcut_path, index=False)

    stored_attributions = pd.DataFrame({
        "id": [1, 2],
        "faithfulness_deletion_delta": [0.1, 0.2],
        "candidate_id": ["deberta-v1", "deberta-v1"],
        "tokenizer_id": ["deberta-tokenizer-v1", "deberta-tokenizer-v1"],
        "checkpoint_id": ["checkpoint-1000", "checkpoint-1000"]
    })
    attributions_path = tmp_path / "stored_attributions.csv"
    stored_attributions.to_csv(attributions_path, index=False)

    notebook = read_notebook("03_final_model_audit.ipynb")
    
    namespace = {
        "prediction_log_path": str(prediction_log_path),
        "artifact_manifest_path": str(manifest_path),
        "probe_predictions_path": str(probe_path),
        "stress_probe_predictions_path": str(stress_path),
        "shortcut_probe_predictions_path": str(shortcut_path),
        "stored_attributions_path": str(attributions_path),
        "candidate_id": "deberta-v1",
        "tokenizer_id": "deberta-tokenizer-v1",
        "checkpoint_id": "checkpoint-1000",
        "evaluation_split": "validation",
        "pd": pd,
        "json": json,
        "Path": Path,
        "display": lambda x: None,
    }

    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(cell.source, namespace)
            if (
                "prediction_log_path = None" in cell.source
                or "prediction_log_path" in cell.metadata.get("tags", [])
            ):
                namespace["prediction_log_path"] = str(prediction_log_path)
                namespace["artifact_manifest_path"] = str(manifest_path)
                namespace["probe_predictions_path"] = str(probe_path)
                namespace["stress_probe_predictions_path"] = str(stress_path)
                namespace["shortcut_probe_predictions_path"] = str(shortcut_path)
                namespace["stored_attributions_path"] = str(attributions_path)
                namespace["candidate_id"] = "deberta-v1"
                namespace["tokenizer_id"] = "deberta-tokenizer-v1"
                namespace["checkpoint_id"] = "checkpoint-1000"
                namespace["evaluation_split"] = "validation"

    assert namespace.get("release_decision") == "submit"
    assert namespace.get("prediction_log") is not None
    
    perf = namespace.get("performance_evidence")
    assert isinstance(perf, pd.DataFrame)
    assert "mae" in perf.columns
    
    rob = namespace.get("robustness_evidence")
    assert isinstance(rob, pd.DataFrame)
    assert "mean_absolute_prediction_delta" in rob.columns
    
    str_ev = namespace.get("stress_evidence")
    assert isinstance(str_ev, pd.DataFrame)
    assert "mean_absolute_prediction_delta" in str_ev.columns
    
    sh_ev = namespace.get("shortcut_evidence")
    assert isinstance(sh_ev, pd.DataFrame)
    assert "mean_absolute_prediction_delta" in sh_ev.columns
    
    x_ev = namespace.get("xai_evidence")
    assert isinstance(x_ev, pd.DataFrame)
    assert "mean_faithfulness_deletion_delta" in x_ev.columns


def test_final_model_audit_execution_with_missing_artifacts(tmp_path: Path) -> None:
    notebook = read_notebook("03_final_model_audit.ipynb")
    
    namespace = {
        "prediction_log_path": None,
        "artifact_manifest_path": None,
        "probe_predictions_path": None,
        "stress_probe_predictions_path": None,
        "shortcut_probe_predictions_path": None,
        "stored_attributions_path": None,
        "candidate_id": None,
        "tokenizer_id": None,
        "checkpoint_id": None,
        "evaluation_split": "validation",
        "pd": pd,
        "json": json,
        "Path": Path,
        "display": lambda x: None,
    }
    
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(cell.source, namespace)
            if "RESULTS_DIR =" in cell.source:
                namespace["RESULTS_DIR"] = tmp_path

    inventory = namespace.get("artifact_inventory")
    assert inventory is not None
    assert (inventory["status"] == "needs-new-evidence").all()
    assert namespace.get("release_decision") == "needs-new-evidence"
