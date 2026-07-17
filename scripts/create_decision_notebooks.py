"""Build deterministic, review-only decision notebooks for the project."""

from pathlib import Path

import nbformat as nbf

# ruff: noqa: E501


ROOT_SETUP = """from pathlib import Path

import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    if (parent / \"pyproject.toml\").is_file()
)
"""

MODEL_REVIEW_SETUP = ROOT_SETUP.replace(
    "from pathlib import Path\n\nimport pandas as pd",
    "import json\nfrom pathlib import Path\n\nimport pandas as pd",
)

DATA_CONTRACT_SETUP = ROOT_SETUP.replace(
    "import pandas as pd", "import pandas as pd\nimport yaml"
)


def observation_template() -> str:
    return """**Observation:** Results are populated only after execution.

**Interpretation:** Do not infer a decision before the displayed evidence is available.

**Action:** Record the evidence, its limitation, and the next decision-layer action."""


def section(heading: str, detail: str) -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(f"{heading}\n\n{detail}"),
        nbf.v4.new_markdown_cell(observation_template()),
    ]


def notebook(filename: str, cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    result = nbf.v4.new_notebook()
    result.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    result.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    for index, cell in enumerate(cells):
        cell["id"] = f"{filename.removesuffix('.ipynb')}-{index:03d}"
        if cell.cell_type == "code":
            cell.source = f"# ruff: noqa: E501\n{cell.source}"
    result.cells = cells
    nbf.validate(result)
    return result


def data_contract_notebook() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Data Contract and EDA\n\nDecision-layer review only; this notebook does not train models."
        ),
        nbf.v4.new_code_cell(
            "data_dir = None\n",
            metadata={"tags": ["parameters"]},
        ),
        nbf.v4.new_code_cell(
            DATA_CONTRACT_SETUP
            + """\nconfig_path = PROJECT_ROOT / \"configs\" / \"default.yaml\"
with config_path.open(encoding=\"utf-8\") as handle:
    data_config = yaml.safe_load(handle)[\"data\"]

if data_dir is None:
    data_dir = PROJECT_ROOT / Path(data_config[\"train_path\"]).parent
else:
    data_dir = Path(data_dir)

def require_csv(filename: str, dtypes: dict[str, str]) -> pd.DataFrame:
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f\"Missing required raw input: {path}\")
    return pd.read_csv(path, dtype=dtypes, encoding=\"ISO-8859-1\")

train = require_csv(
    Path(data_config[\"train_path\"]).name,
    {
        \"id\": \"Int32\",
        \"product_uid\": \"Int32\",
        \"product_title\": \"string\",
        \"search_term\": \"string\",
        \"relevance\": \"float32\",
    },
)
test = require_csv(
    Path(data_config[\"test_path\"]).name,
    {
        \"id\": \"Int32\",
        \"product_uid\": \"Int32\",
        \"product_title\": \"string\",
        \"search_term\": \"string\",
    },
)
product_descriptions = require_csv(
    Path(data_config[\"product_descriptions_path\"]).name,
    {\"product_uid\": \"Int32\", \"product_description\": \"string\"},
)
attributes = require_csv(
    Path(data_config[\"attributes_path\"]).name,
    {\"product_uid\": \"Int32\", \"name\": \"string\", \"value\": \"string\"},
)
"""
        ),
    ]
    cells += section(
        "## 0. Problem Contract",
        "Review relevance-label availability, required raw tables, and decision boundaries.",
    )
    cells += section(
        "## 1. Reproducible Setup",
        "Resolve the project root from the current working directory and use only relative project paths.",
    )
    cells += section(
        "## 2. Raw Data and Join Contract",
        "Validate join keys and row-preservation before combining train, descriptions, and attributes.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """required_train_columns = {\"id\", \"product_uid\", \"search_term\", \"product_title\", \"relevance\"}
missing_train_columns = required_train_columns - set(train.columns)
if missing_train_columns:
    raise ValueError(f\"train.csv is missing columns: {sorted(missing_train_columns)}\")

def require_join_key(table: pd.DataFrame, table_name: str) -> None:
    if \"product_uid\" not in table.columns:
        raise ValueError(f\"{table_name}.product_uid is required for joins.\")
    if not table[\"product_uid\"].notna().all():
        raise ValueError(f\"{table_name}.product_uid must not contain null values.\")

def require_unique_join_key(table: pd.DataFrame, table_name: str) -> None:
    require_join_key(table, table_name)
    if table[\"product_uid\"].duplicated().any():
        raise ValueError(
            f\"{table_name}.product_uid must be unique before joining; \"
            \"deduplicate or aggregate the table first.\"
        )

require_join_key(train, \"train\")
require_unique_join_key(product_descriptions, \"product_descriptions\")
attributes_join = (
    attributes.fillna({\"name\": \"\", \"value\": \"\"})
    .assign(attribute=lambda frame: frame[\"name\"] + \"=\" + frame[\"value\"])
    .groupby(\"product_uid\", as_index=False)[\"attribute\"]
    .agg(\" | \".join)
    .rename(columns={\"attribute\": \"attributes\"})
)
require_unique_join_key(attributes_join, \"attributes\")

joined_train = train.merge(
    product_descriptions, on=\"product_uid\", how=\"left\", validate=\"many_to_one\"
)
if len(joined_train) != len(train):
    raise ValueError(\"Joining product_descriptions changed the train row count.\")
joined_train = joined_train.merge(
    attributes_join, on=\"product_uid\", how=\"left\", validate=\"many_to_one\"
)
if len(joined_train) != len(train):
    raise ValueError(\"Joining attributes changed the train row count.\")

join_contract = pd.DataFrame(
    [
        {
            \"table\": \"product_descriptions\",
            \"key\": \"product_uid\",
            \"key_unique\": product_descriptions[\"product_uid\"].is_unique,
            \"rows_before\": len(train),
            \"rows_after\": len(train.merge(product_descriptions, on=\"product_uid\", how=\"left\")),
        },
        {
            \"table\": \"attributes\",
            \"key\": \"product_uid\",
            \"key_unique\": attributes_join[\"product_uid\"].is_unique,
            \"rows_before\": len(train),
            \"rows_after\": len(joined_train),
        },
    ]
)
join_contract
"""
        )
    )
    cells += section(
        "## 3. Target and Text-Field Audit",
        "Audit target range and null/empty text fields without changing the raw values.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """text_columns = [\"search_term\", \"product_title\"]
target_audit = train[\"relevance\"].describe().to_frame(name=\"relevance\")
text_audit = pd.DataFrame(
    {
        \"null_count\": train[text_columns].isna().sum(),
        \"empty_count\": train[text_columns].fillna(\"\").eq(\"\").sum(),
    }
)
target_audit, text_audit
"""
        )
    )
    cells += section(
        "## 4. Text Quality and Length Profiling",
        "Profile text length and whitespace-token length to set later tokenizer and preprocessing decisions.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """text_profile = pd.DataFrame(
    {
        column: {
            \"characters_p50\": train[column].fillna(\"\").str.len().quantile(0.5),
            \"characters_p99\": train[column].fillna(\"\").str.len().quantile(0.99),
            \"tokens_p99\": train[column].fillna(\"\").str.split().str.len().quantile(0.99),
        }
        for column in text_columns
    }
).T
text_profile
"""
        )
    )
    cells += section(
        "## 5. Duplicate, Near-Duplicate, and Leakage Audit",
        "Use normalized exact-text and row-hash evidence to flag possible leakage for later split decisions.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """normalized_text = train[text_columns].fillna(\"\").apply(
    lambda column: column.str.lower().str.replace(r\"\\s+\", \" \", regex=True).str.strip()
)
normalized_hash = pd.util.hash_pandas_object(normalized_text, index=False)
duplicate_audit = pd.DataFrame(
    {
        \"normalized_exact_duplicate_rows\": [int(normalized_text.duplicated().sum())],
        \"normalized_hash_duplicate_rows\": [int(normalized_hash.duplicated().sum())],
        \"id_overlap_train_test\": [int(train[\"id\"].isin(test[\"id\"]).sum())],
    }
)
duplicate_audit
"""
        )
    )
    cells += section(
        "## 6. Query/Product Dependency and CV Decision",
        "Measure repeated search terms and products; select a fold strategy later only from observed dependence.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """dependency_audit = pd.DataFrame(
    {
        \"unique_search_terms\": [train[\"search_term\"].nunique(dropna=False)],
        \"repeated_search_term_rows\": [int(train[\"search_term\"].duplicated(keep=False).sum())],
        \"unique_products\": [train[\"product_uid\"].nunique(dropna=False)],
        \"repeated_product_rows\": [int(train[\"product_uid\"].duplicated(keep=False).sum())],
    }
)
dependency_audit
"""
        )
    )
    cells += section(
        "## 7. TF-IDF and DeBERTa Tokenizer Readiness",
        "Use whitespace-token distributions as a readiness audit only; no vectorizer or tokenizer is constructed here.",
    )
    cells += section(
        "## 8. Evidence-Based Preprocessing Contract",
        "Document transformations only after the preceding audits justify them; preserve raw text for reproducibility.",
    )
    cells += section(
        "## 9. Findings and Next Actions",
        "Record unresolved data risks and the evidence required before model experimentation.",
    )
    return notebook("01_data_contract_and_eda.ipynb", cells)


def model_review_notebook() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Classical Models and HPO Review\n\nReview persisted artifacts only; no training or search is performed."
        ),
        nbf.v4.new_code_cell(
            "results_dir = None\nselected_experiment = None\nartifact_manifest_path = None\noof_artifact_name = None\nfold_artifact_name = None\nsubmission_artifact_name = None\n",
            metadata={"tags": ["parameters"]},
        ),
        nbf.v4.new_code_cell(
            MODEL_REVIEW_SETUP
            + """\nif results_dir is None:
    results_dir = PROJECT_ROOT / \"results\"
else:
    results_dir = Path(results_dir)

def require_artifact(*candidates: Path) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        \"Missing required experiment artifact. Checked: \"
        + \", \".join(str(path) for path in candidates)
    )

experiment_summary_path = require_artifact(
    results_dir / \"experiment_summary.csv\",
    PROJECT_ROOT / \"logs\" / \"experiments.csv\",
)
hpo_results_path = require_artifact(
    results_dir / \"hpo_results.json\",
    PROJECT_ROOT / \"logs\" / \"hpo\" / \"hpo_results.json\",
)
experiment_summary = pd.read_csv(experiment_summary_path)
with hpo_results_path.open(encoding=\"utf-8\") as handle:
    hpo_results = json.load(handle)
"""
        ),
    ]
    cells += section(
        "## 0. Model-Selection Contract",
        "Compare frozen evidence only. New experiments belong in a separate training workflow.",
    )
    cells += section(
        "## 1. Load Frozen Experiment Artifacts",
        "Require an experiment summary and HPO record; discover OOF, fold, and submission artifacts only when present.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """artifact_directories = (
    results_dir,
    PROJECT_ROOT / \"artifacts\",
    results_dir.parent / \"logs\" / \"hpo\",
    results_dir.parent / \"logs\" / \"runs\",
)

def discover_artifacts(pattern: str) -> list[Path]:
    return sorted(
        (
            path
            for directory in artifact_directories
            if directory.is_dir()
            for path in directory.glob(pattern)
        ),
        key=lambda path: str(path),
    )

oof_candidates = discover_artifacts(\"*_oof.csv\")
fold_candidates = discover_artifacts(\"*_folds.csv\")
submission_candidates = discover_artifacts(\"*submission*.csv\")

def select_artifact(candidates: list[Path], selected_name: str | None) -> Path | None:
    if selected_name is None:
        return None
    matching_candidates = [
        path for path in candidates if selected_name in {path.name, str(path)}
    ]
    return matching_candidates[0] if len(matching_candidates) == 1 else None

def is_selected_artifact_bound(
    selected_name: str | None,
    selected_path: Path | None,
    expected_kind: str,
    manifest: dict[str, object],
) -> bool:
    if selected_name is None:
        needs_new_evidence(f\"an explicit {expected_kind} artifact filename is required.\")
        return False
    if selected_path is None:
        needs_new_evidence(
            f\"selected {expected_kind} artifact is not among discovered candidates: {selected_name}\"
        )
        return False
    manifest_record = manifest.get(selected_name)
    if not isinstance(manifest_record, dict):
        needs_new_evidence(f\"artifact manifest lacks a record for {selected_name}\")
        return False
    if manifest_record.get(\"kind\") != expected_kind:
        needs_new_evidence(f\"artifact manifest kind mismatch for {selected_name}\")
        return False
    if manifest_record.get(\"experiment\") != selected_experiment:
        needs_new_evidence(f\"artifact manifest experiment mismatch for {selected_name}\")
        return False
    return True

optional_artifacts = {
    \"oof\": select_artifact(oof_candidates, oof_artifact_name),
    \"folds\": select_artifact(fold_candidates, fold_artifact_name),
    \"submission\": select_artifact(submission_candidates, submission_artifact_name),
}
{
    \"artifact_directories\": [str(path) for path in artifact_directories],
    \"oof_artifacts\": [str(path) for path in oof_candidates],
    \"fold_artifacts\": [str(path) for path in fold_candidates],
    \"submission_artifacts\": [str(path) for path in submission_candidates],
}
"""
        )
    )
    cells += section(
        "## 2. Validation and Pipeline Boundary Audit",
        "Check that persisted reports document fold-local preprocessing, vectorization, and tokenizer construction requirements.",
    )
    cells += section(
        "## 3. Baseline vs HPO Ridge Comparison",
        "Compare persisted score columns after verifying the metric name and lower-is-better convention.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """if \"rmse\" in experiment_summary.columns:
    metric_column = \"rmse\"
else:
    score_columns = [
        column for column in experiment_summary.columns if \"score\" in column.lower()
    ]
    if not score_columns:
        raise ValueError(
            \"experiment_summary.csv requires an rmse or score metric column\"
        )
    metric_column = score_columns[0]
experiment_summary.sort_values(metric_column).head(10)
if selected_experiment is None:
    selected_experiment = experiment_summary.sort_values(metric_column).iloc[0][\"experiment\"]
"""
        )
    )
    cells += section(
        "## 4. Classical vs DeBERTa Comparison",
        "Compare only architecture labels and persisted metrics; do not load or evaluate model artifacts.",
    )
    cells += section(
        "## 5. Regression Residual and Slice Analysis",
        "Perform residual or slice analysis only when an existing OOF artifact contains label and prediction columns.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """artifact_diagnostics = []

def needs_new_evidence(message: str) -> None:
    diagnostic = f\"needs-new-evidence: {message}\"
    artifact_diagnostics.append(diagnostic)
    print(diagnostic)

def load_artifact_manifest(path: Path | str | None) -> dict[str, object]:
    if path is None:
        needs_new_evidence("artifact manifest is required for OOF/fold analysis.")
        return {}
    manifest_path = Path(path)
    if not manifest_path.is_file():
        needs_new_evidence(f\"artifact manifest is unavailable: {manifest_path}\")
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding=\"utf-8\"))
    except (OSError, json.JSONDecodeError):
        needs_new_evidence(f\"artifact manifest is invalid: {manifest_path}\")
        return {}
    if not isinstance(manifest, dict):
        needs_new_evidence(f\"artifact manifest must map filenames to records: {manifest_path}\")
        return {}
    return manifest

artifact_manifest = load_artifact_manifest(artifact_manifest_path)
oof_path = optional_artifacts[\"oof\"]
oof_is_bound_to_selected_experiment = is_selected_artifact_bound(
    oof_artifact_name, oof_path, \"oof\", artifact_manifest
)
fold_is_bound_to_selected_experiment = is_selected_artifact_bound(
    fold_artifact_name, optional_artifacts[\"folds\"], \"folds\", artifact_manifest
)
if oof_is_bound_to_selected_experiment:
    oof = pd.read_csv(oof_path)
    oof = oof.rename(columns={\"relevance\": \"y_true\", \"prediction\": \"y_pred\"})
    required_oof_columns = {\"y_true\", \"y_pred\"}
    if required_oof_columns.issubset(oof.columns):
        oof = oof.assign(residual=oof[\"y_true\"] - oof[\"y_pred\"])
        oof[[\"y_true\", \"y_pred\", \"residual\"]].describe()
    else:
        raise ValueError(\"OOF artifact requires y_true/y_pred or relevance/prediction columns.\")
"""
        )
    )
    cells += section(
        "## 6. Failure Review",
        "Use persisted worst-case rows or fold summaries when available; otherwise retain the evidence gap.",
    )
    cells += section(
        "## 7. Submission Sanity Checks",
        "Validate an existing submission's exact schema, identifier integrity, and relevance range without writing a file.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """submission_path = optional_artifacts.get(\"submission\")
submission_is_bound_to_selected_experiment = is_selected_artifact_bound(
    submission_artifact_name, submission_path, \"submission\", artifact_manifest
)
if submission_is_bound_to_selected_experiment:
    persisted_submission = pd.read_csv(submission_path)
    expected_submission_columns = [\"id\", \"relevance\"]
    if list(persisted_submission.columns) != expected_submission_columns:
        raise ValueError(
            \"Submission must contain exactly these columns in order: \"
            f\"{expected_submission_columns}; found {list(persisted_submission.columns)}\"
        )
    submission_ids = persisted_submission[\"id\"]
    relevance_values = persisted_submission[\"relevance\"]
    for column, values in ((\"id\", submission_ids), (\"relevance\", relevance_values)):
        if not values.notna().all():
            raise ValueError(f\"Submission {column} must not contain null values.\")
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f\"Submission {column} must be numeric.\")
        if values.isin((float(\"inf\"), float(\"-inf\"))).any():
            raise ValueError(f\"Submission {column} must contain only finite values.\")
    if submission_ids.duplicated().any():
        raise ValueError(\"Submission id values must be unique.\")
    if not relevance_values.between(1.0, 3.0).all():
        raise ValueError(\"Submission relevance values must be in [1.0, 3.0].\")
    submission_checks = pd.DataFrame(
        {
            \"rows\": [len(persisted_submission)],
            \"columns\": [list(persisted_submission.columns)],
            \"id_min\": [submission_ids.min()],
            \"id_max\": [submission_ids.max()],
            \"relevance_min\": [relevance_values.min()],
            \"relevance_max\": [relevance_values.max()],
        }
    )
    submission_checks
"""
        )
    )
    cells += section(
        "## 8. Final Candidate Decision",
        "Set the decision to `submit`, `hold`, or `needs-new-evidence` only after all required artifact checks are complete.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """decision = \"needs-new-evidence\"
decision
"""
        )
    )
    return notebook("02_classical_models_and_hpo_review.ipynb", cells)


def final_audit_notebook() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Final Model Audit\n\nThis audit requires labeled prediction logs; an unlabeled Kaggle submission cannot establish final performance."
        ),
        nbf.v4.new_code_cell(
            'prediction_log_path = None\ncandidate_id = None\ntokenizer_id = None\ncheckpoint_id = None\nevaluation_split = "validation"\n',
            metadata={"tags": ["parameters"]},
        ),
        nbf.v4.new_code_cell(
            ROOT_SETUP
            + """\nif prediction_log_path is None:
    prediction_log_path = PROJECT_ROOT / \"results\" / \"prediction_log.csv\"
else:
    prediction_log_path = Path(prediction_log_path)

if not prediction_log_path.is_file():
    raise FileNotFoundError(f\"Missing prediction log: {prediction_log_path}\")

prediction_log = pd.read_csv(prediction_log_path)

def validate_prediction_log(
    prediction_log: pd.DataFrame,
    candidate_id: str | None,
    tokenizer_id: str | None,
    checkpoint_id: str | None,
    evaluation_split: str = \"validation\",
) -> pd.DataFrame:
    audit_identity = {
        \"candidate_id\": candidate_id,
        \"tokenizer_id\": tokenizer_id,
        \"checkpoint_id\": checkpoint_id,
    }
    for column, value in audit_identity.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f\"{column} parameter must be a non-empty string.\")
    required_columns = {
        \"id\",
        \"split\",
        \"y_true\",
        \"y_pred\",
        \"search_term\",
        \"product_title\",
        \"product_description\",
    } | set(audit_identity)
    missing_columns = required_columns - set(prediction_log.columns)
    if missing_columns:
        raise ValueError(f\"Prediction log is missing columns: {sorted(missing_columns)}\")
    if not prediction_log[\"id\"].notna().all():
        raise ValueError(\"id must not contain null values.\")
    if prediction_log[\"id\"].duplicated().any():
        raise ValueError(\"id values must be unique for final evaluation.\")
    if not isinstance(evaluation_split, str) or not evaluation_split.strip():
        raise ValueError(\"evaluation_split parameter must be a non-empty string.\")
    if evaluation_split not in {\"validation\", \"holdout\", \"test\"}:
        raise ValueError(
            \"evaluation_split must be one of: validation, holdout, or test.\"
        )
    for column, value in audit_identity.items():
        identity_values = prediction_log[column]
        if not identity_values.notna().all():
            raise ValueError(f\"Prediction log {column} must not contain null values.\")
        if not identity_values.eq(value).all():
            raise ValueError(
                f\"Prediction log {column} must equal supplied parameter {value!r} for every row.\"
            )
    if not prediction_log[\"y_true\"].notna().all():
        raise ValueError(\"y_true is required; refusing an unlabeled Kaggle submission.\")
    y_true_values = prediction_log[\"y_true\"]
    if not pd.api.types.is_numeric_dtype(y_true_values):
        raise ValueError(\"y_true must be numeric.\")
    if y_true_values.isin((float(\"inf\"), float(\"-inf\"))).any():
        raise ValueError(\"y_true must contain only finite values.\")
    if not y_true_values.between(1.0, 3.0).all():
        raise ValueError(\"y_true values must be in [1.0, 3.0].\")
    prediction_values = prediction_log[\"y_pred\"]
    if not pd.api.types.is_numeric_dtype(prediction_values):
        raise ValueError(\"y_pred must be numeric.\")
    if not prediction_values.notna().all():
        raise ValueError(\"y_pred must not contain null predictions.\")
    if prediction_values.isin((float(\"inf\"), float(\"-inf\"))).any():
        raise ValueError(\"y_pred must contain only finite predictions.\")
    split_values = prediction_log[\"split\"]
    if not split_values.notna().all() or split_values.astype(str).str.strip().eq(\"\").any():
        raise ValueError(\"split must not contain blank values.\")
    if not split_values.eq(evaluation_split).all():
        raise ValueError(
            \"prediction log rows must all match the permitted evaluation_split \"
            f\"{evaluation_split!r}.\"
        )
    for column in (\"search_term\", \"product_title\", \"product_description\"):
        non_null_text = prediction_log[column].dropna()
        if not non_null_text.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f\"{column} must contain only string values.\")
    for column in (\"search_term\", \"product_title\"):
        if not prediction_log[column].notna().all():
            raise ValueError(f\"{column} must not contain null values.\")
        if prediction_log[column].str.strip().eq(\"\").any():
            raise ValueError(f\"{column} must not contain blank values.\")
    prediction_log = prediction_log.copy()
    prediction_log[\"product_description\"] = prediction_log[
        \"product_description\"
    ].fillna(\"\")
    return prediction_log

prediction_log = validate_prediction_log(
    prediction_log, candidate_id, tokenizer_id, checkpoint_id, evaluation_split
)
metrics = __import__(\"sklearn.metrics\", fromlist=[\"metrics\"])
"""
        ),
    ]
    cells += section(
        "## 0. Audit Contract and Frozen Candidate",
        "Audit one frozen candidate using labeled, persisted predictions. An unlabeled Kaggle submission is not performance evidence.",
    )
    cells += section(
        "## 1. Load Prediction, Split, and Model Artifacts",
        "Load the labeled prediction log; model artifacts are referenced only by recorded metadata, never loaded for inference.",
    )
    cells += section(
        "## 2. Final Performance and Uncertainty",
        "Calculate MAE and RMSE from the validated labeled log and retain split-level uncertainty where available.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """mae = metrics.mean_absolute_error(
    prediction_log[\"y_true\"], prediction_log[\"y_pred\"]
)
rmse = metrics.mean_squared_error(
    prediction_log[\"y_true\"], prediction_log[\"y_pred\"]
) ** 0.5
pd.DataFrame({\"mae\": [mae], \"rmse\": [rmse]})
"""
        )
    )
    cells += section(
        "## 3. Robustness Tests",
        "Validate deterministic text transformations and expected invariance or risk hypotheses only; this notebook does not infer predictions or measure model degradation.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """def normalize_probe_text(text: str) -> str:
    unicode_text = __import__(\"unicodedata\").normalize(\"NFKC\", text)
    return \" \".join(unicode_text.lower().split())

def build_text_probe_table(
    prediction_log: pd.DataFrame, limit: int = 5
) -> pd.DataFrame:
    if limit < 1:
        raise ValueError(\"limit must be at least one.\")
    probe_rows = []
    for _, row in prediction_log.head(limit).iterrows():
        search_term = row[\"search_term\"]
        product_title = row[\"product_title\"]
        product_description = row[\"product_description\"][:200]
        variants = [
            (
                \"case_whitespace_punctuation\",
                f\"  {search_term.upper()}!!!  \",
                product_title,
                product_description,
                \"normalized text should preserve the recorded text meaning\",
            ),
            (
                \"missing_brand_or_attribute\",
                search_term,
                product_title,
                f\"brand=<missing>; attributes=<missing>; {product_description}\",
                \"missing context is a documented risk, not a prediction result\",
            ),
            (
                \"long_description\",
                search_term,
                product_title,
                ((product_description + \" \") * 20).strip(),
                \"truncation policy should be reviewed for long text\",
            ),
            (
                \"empty_description\",
                search_term,
                product_title,
                \"\",
                \"missing-description handling should follow the recorded policy\",
            ),
            (
                \"repeated_description\",
                search_term,
                product_title,
                \" \".join([product_description] * 3).strip(),
                \"repetition should not be mistaken for independent evidence\",
            ),
            (
                \"keyword_stuffing\",
                search_term,
                f\"{search_term} {search_term} {search_term} {product_title}\",
                product_description,
                \"keyword frequency is a shortcut-risk hypothesis\",
            ),
            (
                \"title_description_contradiction\",
                search_term,
                f\"NOT {product_title}\",
                f\"Contradicts title: {product_description}\",
                \"title-description conflict is a shortcut-risk hypothesis\",
            ),
        ]
        for scenario, probe_query, probe_title, probe_description, hypothesis in variants:
            probe_rows.append(
                {
                    \"id\": row[\"id\"],
                    \"scenario\": scenario,
                    \"search_term\": normalize_probe_text(probe_query),
                    \"product_title\": normalize_probe_text(probe_title),
                    \"product_description\": normalize_probe_text(probe_description),
                    \"expected_hypothesis\": hypothesis,
                }
            )
    return pd.DataFrame(probe_rows)

probe_table = build_text_probe_table(prediction_log, limit=5)
probe_table
"""
        )
    )
    cells += section(
        "## 4. Stress Tests",
        "Review deterministic long-text, empty-description, and sparse-attribute scenarios without generating predictions.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """stress_probe_table = probe_table[
    probe_table[\"scenario\"].isin(
        [\"long_description\", \"empty_description\", \"repeated_description\"]
    )
]
stress_probe_table
"""
        )
    )
    cells += section(
        "## 5. Adversarial / Shortcut Tests",
        "Review shortcut-risk scenarios such as identifier dependence and repeated product text without inference.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """shortcut_probe_table = probe_table[
    probe_table[\"scenario\"].isin(
        [\"keyword_stuffing\", \"title_description_contradiction\"]
    )
]
shortcut_probe_table
"""
        )
    )
    cells += section(
        "## 6. Error Slices and Failure Taxonomy",
        "Slice validated residuals by available text-length and split columns, then document failure categories.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """prediction_log = prediction_log.assign(
    residual=prediction_log[\"y_true\"] - prediction_log[\"y_pred\"],
    absolute_error=lambda frame: frame[\"residual\"].abs(),
    query_length_bin=pd.cut(
        prediction_log[\"search_term\"].str.len(), [0, 5, 15, 30, float(\"inf\")]
    ),
    description_presence=prediction_log[\"product_description\"].ne(\"\"),
    relevance_band=pd.cut(prediction_log[\"y_true\"], [0, 1.5, 2.5, float(\"inf\")]),
)
slice_dimensions = [\"query_length_bin\", \"description_presence\", \"split\", \"relevance_band\"]
error_slices = pd.concat(
    [
        prediction_log.groupby(dimension, dropna=False).agg(
            count=(\"id\", \"size\"),
            mae=(\"absolute_error\", \"mean\"),
            rmse=(\"residual\", lambda values: (values.pow(2).mean()) ** 0.5),
            mean_residual=(\"residual\", \"mean\"),
        ).reset_index().assign(slice_dimension=dimension)
        for dimension in slice_dimensions
    ],
    ignore_index=True,
)
failure_taxonomy = prediction_log.assign(
    failure_taxonomy=lambda frame: frame.apply(
        lambda row: (
            \"high_error_missing_description\"
            if row[\"absolute_error\"] >= frame[\"absolute_error\"].quantile(0.9)
            and not row[\"description_presence\"]
            else \"high_error_long_query\"
            if row[\"absolute_error\"] >= frame[\"absolute_error\"].quantile(0.9)
            and len(row[\"search_term\"]) > 30
            else \"high_error_extreme_relevance\"
            if row[\"absolute_error\"] >= frame[\"absolute_error\"].quantile(0.9)
            and (row[\"y_true\"] <= 1.5 or row[\"y_true\"] >= 2.5)
            else \"other\"
        ),
        axis=1,
    )
)
failure_taxonomy_summary = failure_taxonomy.groupby(\"failure_taxonomy\", dropna=False).agg(
    count=(\"id\", \"size\"), mean_absolute_error=(\"absolute_error\", \"mean\")
).reset_index()
top_absolute_error_rows = prediction_log.nlargest(20, \"absolute_error\")
taxonomy_top_error_rows = failure_taxonomy.nlargest(20, \"absolute_error\")
error_slices, taxonomy_top_error_rows, failure_taxonomy_summary
"""
        )
    )
    cells += section(
        "## 7. XAI and Explanation Validation",
        "XAI is limited to stored attribution data when present; no model is loaded, trained, or used for inference.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """attribution_path = prediction_log_path.with_name(\"stored_attributions.csv\")
if attribution_path.is_file():
    stored_attributions = pd.read_csv(attribution_path)
    required_attribution_columns = {
        \"candidate_id\",
        \"tokenizer_id\",
        \"checkpoint_id\",
        \"id\",
        \"faithfulness_deletion_delta\",
    }
    missing_attribution_columns = required_attribution_columns - set(stored_attributions.columns)
    if missing_attribution_columns:
        raise ValueError(f\"Stored attributions are missing: {sorted(missing_attribution_columns)}\")
    attribution_identity = {
        \"candidate_id\": candidate_id,
        \"tokenizer_id\": tokenizer_id,
        \"checkpoint_id\": checkpoint_id,
    }
    for column, value in attribution_identity.items():
        mismatched_rows = stored_attributions.loc[
            ~stored_attributions[column].eq(value), column
        ]
        if not mismatched_rows.empty:
            raise ValueError(
                f\"Stored attributions contain {len(mismatched_rows)} rows with mismatching \"
                f\"{column}: {mismatched_rows.head(10).tolist()}\"
            )
    unlinked_ids = stored_attributions.loc[
        ~stored_attributions[\"id\"].isin(prediction_log[\"id\"]), \"id\"
    ]
    if not stored_attributions[\"id\"].isin(prediction_log[\"id\"]).all():
        raise ValueError(
            f\"Stored attributions contain {len(unlinked_ids)} unlinked IDs: {unlinked_ids.head(10).tolist()}\"
        )
    faithfulness_values = stored_attributions[\"faithfulness_deletion_delta\"]
    if not pd.api.types.is_numeric_dtype(faithfulness_values) or not faithfulness_values.notna().all():
        raise ValueError(\"Stored attributions require numeric faithfulness_deletion_delta values.\")
    if faithfulness_values.isin((float(\"inf\"), float(\"-inf\"))).any() or not (faithfulness_values > 0).all():
        raise ValueError(\"faithfulness_deletion_delta > 0 is required for faithfulness validation.\")
    print(\"Positive faithfulness_deletion_delta means masking/deletion worsened the prediction objective.\")
    stored_attributions
else:
    xai_evidence_state = \"XAI evidence unavailable\"
    print(xai_evidence_state)
"""
        )
    )
    cells += section(
        "## 8. Release Decision and Known Limitations",
        "Set `submit`, `hold`, or `remediate` from the executed audit evidence and preserve known limitations.",
    )
    cells.append(
        nbf.v4.new_code_cell(
            """release_decision = \"hold\"
release_decision
"""
        )
    )
    return notebook("03_final_model_audit.ipynb", cells)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebooks_dir = project_root / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)
    generated_notebooks = {
        "03_final_model_audit.ipynb": final_audit_notebook(),
    }
    for filename, generated_notebook in generated_notebooks.items():
        nbf.write(generated_notebook, notebooks_dir / filename)


if __name__ == "__main__":
    main()
