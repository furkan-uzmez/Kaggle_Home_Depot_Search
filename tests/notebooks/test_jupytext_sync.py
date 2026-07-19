import builtins
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import jupytext
import nbformat
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ONE_SOURCE = PROJECT_ROOT / "notebooks" / "01_data_contract_and_eda.py"
NOTEBOOK_ONE = PROJECT_ROOT / "notebooks" / "01_data_contract_and_eda.ipynb"
NOTEBOOK_TWO_SOURCE = (
    PROJECT_ROOT / "notebooks" / "02_classical_models_and_hpo_review.py"
)
NOTEBOOK_TWO = PROJECT_ROOT / "notebooks" / "02_classical_models_and_hpo_review.ipynb"
NOTEBOOK_THREE_SOURCE = PROJECT_ROOT / "notebooks" / "03_final_model_audit.py"
NOTEBOOK_THREE = PROJECT_ROOT / "notebooks" / "03_final_model_audit.ipynb"
NOTEBOOK_ONE_HEADINGS = (
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
)
INSTRUCTIONAL_PLACEHOLDERS = (
    "Fill in the observed facts from the preceding executed output.",
    "Explain what the observed facts mean for this analysis.",
    "State the next action justified by the preceding output.",
    "Do not invent metrics.",
)
NOTEBOOK_TWO_PLACEHOLDERS = (
    "Results are populated only after execution.",
    "Do not infer a decision before the displayed evidence is available.",
    "Record the evidence, its limitation, and the next decision-layer action.",
)
NOTEBOOK_TWO_HEADINGS = (
    "## 0. Model-Selection Contract",
    "## 1. Load Frozen Experiment Artifacts",
    "## 2. Validation and Pipeline Boundary Audit",
    "## 3. Baseline vs HPO Ridge Comparison",
    "## 4. Classical vs DeBERTa Comparison",
    "## 5. Regression Residual and Slice Analysis",
    "## 6. Failure Review",
    "## 7. Submission Sanity Checks",
    "## 8. Final Candidate Decision",
)
NOTEBOOK_THREE_PLACEHOLDERS = (
    "Results are populated only after execution.",
    "Do not infer a decision before the displayed evidence is available.",
    "Record the evidence, its limitation, and the next decision-layer action.",
)
NOTEBOOK_THREE_HEADINGS = (
    "## 0. Audit Contract and Frozen Candidate",
    "## 1. Load Prediction, Split, and Model Artifacts",
    "## 2. Final Performance and Uncertainty",
    "## 3. Robustness Tests",
    "## 4. Stress Tests",
    "## 5. Adversarial / Shortcut Tests",
    "## 6. Error Slices and Failure Taxonomy",
    "## 7. XAI and Explanation Validation",
    "## 8. Release Decision and Known Limitations",
)
ANALYSIS_CELL_MARKERS = (
    "join_contract",
    "target_audit, text_audit",
    "text_profile",
    "duplicate_audit",
    "dependency_audit",
    "tokenizer_length_percentiles, truncation_metrics",
    "preprocessing_contract",
)


def test_jupytext_config_preserves_notebook_and_parameter_metadata() -> None:
    with (PROJECT_ROOT / "jupytext.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    assert config["formats"] == "ipynb,py:percent"
    assert config["notebook_metadata_filter"] == "kernelspec,jupytext"
    assert (
        config["cell_metadata_filter"]
        == "all,-autoscroll,-collapsed,-scrolled,-trusted,-ExecuteTime,-execution"
    )


def test_jupytext_round_trips_paired_notebook_metadata() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
        fixture_dir = Path(temporary_directory)
        notebook_path = fixture_dir / "fixture.ipynb"
        notebook = nbformat.v4.new_notebook(
            metadata={
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                }
            },
            cells=[
                nbformat.v4.new_code_cell(
                    "limit = 5",
                    metadata={
                        "tags": ["parameters"],
                        "custom_metadata": "preserve-me",
                    },
                )
            ],
        )
        nbformat.write(notebook, notebook_path)

        pair_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupytext",
                "--set-formats",
                "ipynb,py:percent",
                str(notebook_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert pair_result.returncode == 0, pair_result.stderr
        source_path = notebook_path.with_suffix(".py")
        paired_source = source_path.read_text(encoding="utf-8")
        assert 'custom_metadata="preserve-me"' in paired_source

        output_path = fixture_dir / "round_trip.ipynb"
        convert_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupytext",
                "--to",
                "ipynb",
                "--output",
                str(output_path),
                str(source_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert convert_result.returncode == 0, convert_result.stderr
        round_trip = nbformat.read(output_path, as_version=4)

    assert round_trip.cells[0].metadata["tags"] == ["parameters"]
    assert round_trip.cells[0].metadata["custom_metadata"] == "preserve-me"
    assert round_trip.metadata["kernelspec"] == {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }


def test_notebook_one_percent_source_has_executable_evidence_contract() -> None:
    assert NOTEBOOK_ONE_SOURCE.is_file()
    source = NOTEBOOK_ONE_SOURCE.read_text(encoding="utf-8")

    for heading in NOTEBOOK_ONE_HEADINGS:
        assert heading in source

    assert "combined_text" in source
    assert "local_files_only=True" in source
    assert "truncation_count" in source
    assert "truncation_rate" in source
    assert "needs-new-evidence" in source

    notebook = jupytext.read(NOTEBOOK_ONE_SOURCE)
    for heading in NOTEBOOK_ONE_HEADINGS:
        heading_index = next(
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "markdown" and heading in cell.source
        )
        next_heading_index = next(
            (
                index
                for index, cell in enumerate(
                    notebook.cells[heading_index + 1 :],
                    heading_index + 1,
                )
                if cell.cell_type == "markdown" and cell.source.startswith("## ")
            ),
            len(notebook.cells),
        )
        section_code = notebook.cells[heading_index + 1 : next_heading_index]
        assert any(
            cell.cell_type == "code" and cell.source.strip() for cell in section_code
        )


def test_notebook_one_ipynb_matches_percent_source() -> None:
    assert NOTEBOOK_ONE.is_file()
    expected_notebook = jupytext.read(NOTEBOOK_ONE_SOURCE)
    checked_in_notebook = nbformat.read(NOTEBOOK_ONE, as_version=4)

    def notebook_signature(notebook: nbformat.NotebookNode) -> tuple[object, object]:
        cells = [
            (
                cell.cell_type,
                cell.source,
                {
                    key: value
                    for key, value in cell.metadata.items()
                    if key != "execution"
                },
            )
            for cell in notebook.cells
        ]
        metadata = dict(notebook.metadata)
        jupytext_metadata = dict(metadata.get("jupytext", {}))
        jupytext_metadata.pop("text_representation", None)
        metadata["jupytext"] = jupytext_metadata
        metadata.pop("language_info", None)
        return metadata, cells

    assert notebook_signature(checked_in_notebook) == notebook_signature(
        expected_notebook
    )


def test_notebook_two_ipynb_matches_percent_source() -> None:
    assert NOTEBOOK_TWO_SOURCE.is_file()
    assert NOTEBOOK_TWO.is_file()

    expected_notebook = jupytext.read(NOTEBOOK_TWO_SOURCE)
    checked_in_notebook = nbformat.read(NOTEBOOK_TWO, as_version=4)

    def notebook_signature(notebook: nbformat.NotebookNode) -> tuple[object, object]:
        cells = [
            (
                cell.cell_type,
                cell.source,
                {
                    key: value
                    for key, value in cell.metadata.items()
                    if key != "execution"
                },
            )
            for cell in notebook.cells
        ]
        metadata = dict(notebook.metadata)
        jupytext_metadata = dict(metadata.get("jupytext", {}))
        jupytext_metadata.pop("text_representation", None)
        metadata["jupytext"] = jupytext_metadata
        metadata.pop("language_info", None)
        return metadata, cells

    assert notebook_signature(checked_in_notebook) == notebook_signature(
        expected_notebook
    )


def test_notebook_two_markdown_uses_saved_evidence_not_instructional_placeholders() -> (
    None
):
    source = NOTEBOOK_TWO_SOURCE.read_text(encoding="utf-8")
    notebook = jupytext.read(NOTEBOOK_TWO_SOURCE)
    markdown_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    generated_notebook = nbformat.read(NOTEBOOK_TWO, as_version=4)
    generated_markdown = "\n".join(
        cell.source for cell in generated_notebook.cells if cell.cell_type == "markdown"
    )

    for placeholder in NOTEBOOK_TWO_PLACEHOLDERS:
        assert placeholder not in source
        assert placeholder not in generated_markdown

    assert generated_markdown == markdown_source
    assert markdown_source.count("**Observation:**") == 9
    assert markdown_source.count("**Interpretation:**") == 9
    assert markdown_source.count("**Action:**") == 9
    assert markdown_source.count("manifest-bound") >= 5


def test_notebook_two_places_executable_evidence_before_each_oia_note() -> None:
    notebook = jupytext.read(NOTEBOOK_TWO_SOURCE)

    for heading in NOTEBOOK_TWO_HEADINGS:
        heading_index = next(
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "markdown" and heading in cell.source
        )
        next_heading_index = next(
            (
                index
                for index, cell in enumerate(
                    notebook.cells[heading_index + 1 :], heading_index + 1
                )
                if cell.cell_type == "markdown" and cell.source.startswith("## ")
            ),
            len(notebook.cells),
        )
        section_cells = notebook.cells[heading_index + 1 : next_heading_index]

        assert len(section_cells) >= 2
        assert section_cells[0].cell_type == "code"
        assert (
            "display(" in section_cells[0].source
            or "display_table(" in section_cells[0].source
            or "display as show_" in section_cells[0].source
        )
        decision_note = section_cells[-1]
        assert decision_note.cell_type == "markdown"
        assert "**Observation:**" in decision_note.source
        assert "**Interpretation:**" in decision_note.source
        assert "**Action:**" in decision_note.source


def test_notebook_two_uses_executed_artifact_tables_not_static_numeric_claims() -> None:
    notebook = jupytext.read(NOTEBOOK_TWO_SOURCE)
    markdown_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    source = NOTEBOOK_TWO_SOURCE.read_text(encoding="utf-8")

    assert "baseline_hpo_comparison" in source
    assert "decision_inputs" in source
    assert "artifact_inventory" in source
    assert "available_path" in source
    assert "missing_provenance_condition" in source
    assert "required_action" in source
    for static_value in (
        "0.5219995884667691",
        "0.5219632744772689",
        "10.620869580141791",
        'decision = "needs-new-evidence"',
    ):
        assert static_value not in markdown_source
        assert static_value not in source


def test_notebook_three_migrates_to_jupytext_with_concrete_artifact_evidence() -> None:
    assert NOTEBOOK_THREE_SOURCE.is_file()
    assert NOTEBOOK_THREE.is_file()

    source = NOTEBOOK_THREE_SOURCE.read_text(encoding="utf-8")
    notebook = jupytext.read(NOTEBOOK_THREE_SOURCE)
    markdown_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    generated_notebook = nbformat.read(NOTEBOOK_THREE, as_version=4)
    generated_markdown = "\n".join(
        cell.source for cell in generated_notebook.cells if cell.cell_type == "markdown"
    )

    for placeholder in NOTEBOOK_THREE_PLACEHOLDERS:
        assert placeholder not in source
        assert placeholder not in markdown_source
        assert placeholder not in generated_markdown

    assert generated_markdown == markdown_source
    assert markdown_source.count("**Observation:**") == 9
    assert markdown_source.count("**Interpretation:**") == 9
    assert markdown_source.count("**Action:**") == 9
    assert "artifact_inventory" in source

    audit_artifact_paths = (
        PROJECT_ROOT / "results" / "prediction_log.csv",
        PROJECT_ROOT / "results" / "artifact_manifest.json",
        PROJECT_ROOT / "results" / "probe_predictions.csv",
        PROJECT_ROOT / "results" / "stress_probe_predictions.csv",
        PROJECT_ROOT / "results" / "shortcut_probe_predictions.csv",
        PROJECT_ROOT / "results" / "stored_attributions.csv",
    )

    for path in audit_artifact_paths:
        assert path.name in source


def test_notebook_three_renders_artifact_inventory_before_audit_headings() -> None:
    notebook = jupytext.read(NOTEBOOK_THREE_SOURCE)
    first_heading_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown"
        and "## 0. Audit Contract and Frozen Candidate" in cell.source
    )
    inventory_cells = notebook.cells[:first_heading_index]

    assert any(
        cell.cell_type == "code"
        and "artifact_inventory" in cell.source
        and "display(" in cell.source
        for cell in inventory_cells
    )


def test_notebook_three_orders_real_evidence_before_non_static_oia_notes() -> None:
    notebook = jupytext.read(NOTEBOOK_THREE_SOURCE)
    source = NOTEBOOK_THREE_SOURCE.read_text(encoding="utf-8")

    assert 'release_decision = "needs-new-evidence"' not in source
    assert "build_text_probe_table" not in source
    assert "expected_hypothesis" not in source

    for heading in NOTEBOOK_THREE_HEADINGS:
        heading_index = next(
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "markdown" and heading in cell.source
        )
        next_heading_index = next(
            (
                index
                for index, cell in enumerate(
                    notebook.cells[heading_index + 1 :], heading_index + 1
                )
                if cell.cell_type == "markdown" and cell.source.startswith("## ")
            ),
            len(notebook.cells),
        )
        section_cells = notebook.cells[heading_index + 1 : next_heading_index]

        assert len(section_cells) >= 2
        assert section_cells[0].cell_type == "code"
        assert "display(" in section_cells[0].source
        assert "needs-new-evidence" in section_cells[0].source
        decision_note = section_cells[-1]
        assert decision_note.cell_type == "markdown"
        assert "**Observation:**" in decision_note.source
        assert "**Interpretation:**" in decision_note.source
        assert "**Action:**" in decision_note.source
        assert "preceding" in decision_note.source.lower()


def test_notebook_three_ipynb_matches_percent_source() -> None:
    expected_notebook = jupytext.read(NOTEBOOK_THREE_SOURCE)
    checked_in_notebook = nbformat.read(NOTEBOOK_THREE, as_version=4)

    def notebook_signature(notebook: nbformat.NotebookNode) -> tuple[object, object]:
        cells = [
            (
                cell.cell_type,
                cell.source,
                {
                    key: value
                    for key, value in cell.metadata.items()
                    if key != "execution"
                },
            )
            for cell in notebook.cells
        ]
        metadata = dict(notebook.metadata)
        jupytext_metadata = dict(metadata.get("jupytext", {}))
        jupytext_metadata.pop("text_representation", None)
        metadata["jupytext"] = jupytext_metadata
        metadata.pop("language_info", None)
        return metadata, cells

    assert notebook_signature(checked_in_notebook) == notebook_signature(
        expected_notebook
    )


def tokenizer_readiness_helper():
    notebook = jupytext.read(NOTEBOOK_ONE_SOURCE)
    readiness_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "def build_tokenizer_readiness" in cell.source
    )
    namespace = {"Any": Any, "Path": Path, "TypedDict": TypedDict, "pd": pd}
    exec(readiness_cell, namespace)
    return namespace["build_tokenizer_readiness"]


def test_notebook_one_places_concrete_evidence_notes_after_each_analysis_cell() -> None:
    source = NOTEBOOK_ONE_SOURCE.read_text(encoding="utf-8")
    for placeholder in INSTRUCTIONAL_PLACEHOLDERS:
        assert placeholder not in source

    notebook = jupytext.read(NOTEBOOK_ONE_SOURCE)

    for marker in ANALYSIS_CELL_MARKERS:
        analysis_index = next(
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code" and cell.source.strip().endswith(marker)
        )
        next_heading_index = next(
            (
                index
                for index, cell in enumerate(
                    notebook.cells[analysis_index + 1 :], analysis_index + 1
                )
                if cell.cell_type == "markdown" and cell.source.startswith("## ")
            ),
            len(notebook.cells),
        )
        analyst_note = notebook.cells[next_heading_index - 1]

        assert analyst_note.cell_type == "markdown"
        assert "### Observation" in analyst_note.source
        assert "### Interpretation" in analyst_note.source
        assert "### Action" in analyst_note.source
        for placeholder in INSTRUCTIONAL_PLACEHOLDERS:
            assert placeholder not in analyst_note.source


def test_notebook_one_keeps_decision_notes_out_of_code() -> None:
    source = NOTEBOOK_ONE_SOURCE.read_text(encoding="utf-8")

    assert "decision_note" not in source


def test_tokenizer_readiness_reports_missing_local_evidence() -> None:
    def unavailable_loader(*args: object, **kwargs: object) -> object:
        raise ImportError("transformers is unavailable")

    readiness = tokenizer_readiness_helper()(
        pd.Series(["small text"]),
        "local-tokenizer",
        unavailable_loader,
    )

    assert readiness["truncation_metrics"].empty
    assert readiness["diagnostics"] == [
        "needs-new-evidence: local DeBERTa tokenizer is unavailable for "
        "'local-tokenizer': transformers is unavailable"
    ]


def test_tokenizer_readiness_handles_unavailable_transformers_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def unavailable_transformers_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> object:
        if name == "transformers":
            raise ImportError("transformers is unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", unavailable_transformers_import)

    readiness = tokenizer_readiness_helper()(
        pd.Series(["small text"]), "local-tokenizer"
    )

    assert readiness["truncation_metrics"].empty
    assert readiness["diagnostics"] == [
        "needs-new-evidence: local DeBERTa tokenizer is unavailable for "
        "'local-tokenizer': transformers is unavailable"
    ]


def test_tokenizer_readiness_reports_truncation_metrics_from_stub_tokenizer() -> None:
    class StubTokenizer:
        def __call__(
            self,
            texts: list[str],
            *,
            add_special_tokens: bool,
            truncation: bool,
            padding: bool,
        ) -> dict[str, list[list[int]]]:
            assert add_special_tokens is True
            assert truncation is False
            assert padding is False
            return {
                "input_ids": [[0] * (64 if text == "short" else 513) for text in texts]
            }

    loader_calls: list[tuple[object, object]] = []

    def local_loader(*args: object, **kwargs: object) -> StubTokenizer:
        loader_calls.append((args, kwargs.get("local_files_only")))
        return StubTokenizer()

    readiness = tokenizer_readiness_helper()(
        pd.Series(["short", "long"]),
        "local-tokenizer",
        local_loader,
    )

    assert loader_calls == [(("local-tokenizer",), True)]
    assert readiness["diagnostics"] == []
    assert readiness["truncation_metrics"].to_dict("records") == [
        {"max_length": 128, "truncation_count": 1, "truncation_rate": 0.5},
        {"max_length": 256, "truncation_count": 1, "truncation_rate": 0.5},
        {"max_length": 512, "truncation_count": 1, "truncation_rate": 0.5},
    ]
