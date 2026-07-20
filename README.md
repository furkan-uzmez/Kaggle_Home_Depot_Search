# Home Depot Search Relevance

A reproducible, leakage-aware pipeline for the [Kaggle Home Depot Product Search Relevance](https://www.kaggle.com/c/home-depot-product-search-relevance) competition: predict how relevant a product is to a search query (regression, 1.0–3.0) from query/title/description/attribute text.

What sets this repo apart from a typical competition script dump: every modeling claim is backed by a **jupytext-paired decision notebook** that refuses to state a conclusion — a metric, a ranking, a release decision — without a persisted, manifest-bound artifact behind it. Notebooks that lack evidence print `needs-new-evidence` instead of guessing.

## Features

- **Leakage-aware CV** — relevance-stratified folds, fold-local feature fitting (TF-IDF/SVD/scalers fit on the training fold only), duplicate/entity-overlap auditing.
- **Feature + model registry** — TF-IDF/SVD variants (full text, search term, title), a heuristic text-overlap feature set, Ridge and mean/median baselines, all pluggable via `run_experiments.py`.
- **Optuna HPO** for the Ridge + TF-IDF/SVD pipeline.
- **Optional DeBERTa fine-tuning** (`microsoft/deberta-v3-small`) as a transformer baseline.
- **Evidence-gated decision notebooks** — EDA/data-contract audit, classical-model/HPO review, and a final-model audit (robustness, stress, and shortcut probes plus deletion-based attribution faithfulness), each with Observation → Interpretation → Action notes tied to real, persisted output.
- **Experiment logging** to `logs/experiments.csv` / `artifacts/*.json`, bound to notebook evidence through `results/artifact_manifest.json`.

## Requirements

- Python 3.11 (see `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for dependency management and running commands
- Kaggle competition data (see [Data](#data)) — not included, not redistributable
- Optional: a GPU speeds up DeBERTa fine-tuning; CPU works for everything else

## Installation

```bash
uv sync
```

Installs the runtime + `dev` dependency group (pandas/scikit-learn stack, pytest, ruff, jupytext, nbconvert). This is enough for the classical pipeline, all notebooks except DeBERTa-specific cells, and most of the test suite.

The optional group (`torch`, `transformers`, `optuna`, `xgboost`, `sentence-transformers` — several GB) is only needed for HPO, DeBERTa fine-tuning, and the full test suite:

```bash
uv sync --all-groups
```

> Running `uv run pytest` with only the base install will abort at collection (`ModuleNotFoundError: No module named 'torch'`) because two test modules import the optional group. Use `--all-groups`, or scope the run — see [Troubleshooting](#troubleshooting).

## Data

Download the competition files from Kaggle and place them under `data/`:

- `train.csv`
- `test.csv`
- `attributes.csv`
- `product_descriptions.csv`

`data/` is git-ignored — the dataset is under Kaggle's competition license and must not be committed or redistributed.

## Quick Start

Proof of life, no data required:

```bash
uv sync
uv run python run_experiments.py --list-features
```

Expected output: a table of registered feature sets (`baseline-mean`, `tfidf-svd`, `tfidf-svd-search-term`, `tfidf-svd-product-title`, `tfidf-svd-all-text`, `text-overlap-v2`).

With the data in place, run a leakage-aware 5-fold CV experiment:

```bash
uv run python run_experiments.py --features tfidf-svd text-overlap-v2 --models ridge
```

Expected output: a per feature/model RMSE table printed to the console, with matching rows appended to `logs/experiments.csv` and per-run JSON under `artifacts/`.

## Usage

List what's available:

```bash
uv run python run_experiments.py --list-features
uv run python run_experiments.py --list-models
```

Run EDA and a sequence-length report (writes to `outputs/reports/`):

```bash
uv run python run_experiments.py --run-eda
```

Hyperparameter search for Ridge + TF-IDF/SVD (writes `logs/hpo/hpo_results.json`):

```bash
uv run python run_experiments.py --hpo --hpo-trials 20 --hpo-folds 3
```

Generate a Kaggle submission from a trained feature/model pair:

```bash
uv run python run_experiments.py --make-submission --submission-feature tfidf-svd --submission-model ridge
```

Fine-tune the optional DeBERTa baseline (requires `uv sync --all-groups`, GPU recommended):

```bash
uv run python run_experiments.py --transformer-train --transformer-epochs 3
```

Validate the raw data files against the expected schema and write a checksummed manifest to `artifacts/data_manifest.json`:

```bash
uv run python -m home_depot_search.cli --check-data
```

Generate the persisted evidence the final-audit notebook depends on — a holdout prediction log plus robustness/stress/shortcut probe pairs and attribution faithfulness, via inference only (no training):

```bash
uv run python scripts/generate_audit_artifacts.py
```

## Decision Notebooks

`notebooks/` holds three notebooks that build the case for (or against) shipping a candidate model, each paired 1:1 with a `.py` source in the same directory via [jupytext](https://jupytext.readthedocs.io/) (`ipynb,py:percent`):

| Notebook | Question it answers |
| --- | --- |
| `01_data_contract_and_eda` | Is the raw data, target, and text quality what the pipeline assumes? Duplicate/leakage audit, query/product dependency, tokenizer readiness. |
| `02_classical_models_and_hpo_review` | Which classical candidate wins, and is that ranking backed by manifest-bound OOF/fold/submission artifacts? |
| `03_final_model_audit` | Is the selected candidate safe to release? Held-out performance with bootstrap uncertainty, robustness/stress/shortcut probes, error slices, and explanation faithfulness. |

Every analysis section ends with an **Observation / Interpretation / Action** note computed from the cell directly above it — never a static claim. Sections that need an artifact that hasn't been persisted (an unbound OOF file, a missing prediction log) print a `needs-new-evidence` diagnostic instead of a number.

Edit the `.py` source, then re-sync and re-run:

```bash
uv run jupytext --sync notebooks/01_data_contract_and_eda.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_contract_and_eda.ipynb
```

`notebooks/kaggle_remaining_experiments.ipynb` is an unpaired scratch notebook, not part of the decision-notebook contract.

## Repository Map

| Path | Contents |
| --- | --- |
| `src/home_depot_search/` | Library code: data loading/cleaning, leakage-aware splits, feature/model registry, evaluation, HPO, transformer training, audit-artifact generation. |
| `run_experiments.py` | CLI entry point for experiments, EDA, HPO, submissions, and transformer training. |
| `scripts/` | One-off generators: decision-notebook scaffolding, audit-artifact generation. |
| `notebooks/` | Evidence-gated decision notebooks, jupytext-paired `.ipynb`/`.py`. |
| `configs/default.yaml` | Default data paths, model/eda/hpo/transformer/submission settings. |
| `tests/` | pytest suite mirroring `src/` (plus `tests/notebooks/` for notebook structure and execution checks). |
| `data/` *(git-ignored)* | Kaggle CSVs — see [Data](#data). |
| `logs/`, `artifacts/`, `outputs/`, `results/` *(git-ignored)* | Generated experiment logs, per-run artifacts, reports, and audit outputs. |

## Configuration

Defaults live in `configs/default.yaml`; override via CLI flags rather than editing the file for one-off runs.

| Setting | Default | Description |
| --- | --- | --- |
| `data.*_path` | `data/*.csv` | Raw input locations. |
| `hpo.n_trials` / `hpo.n_folds` | `50` / `3` | Optuna trial and CV-fold counts. |
| `transformer.model_name` | `microsoft/deberta-v3-small` | Base checkpoint for the optional transformer path. |
| `transformer.max_length` / `batch_size` / `epochs` / `learning_rate` | `256` / `16` / `3` / `2e-5` | DeBERTa fine-tuning hyperparameters. |
| `submission.clamp_range` | `[1.0, 3.0]` | Range predictions are clipped to before writing a submission. |

No secrets or API keys are required anywhere in this project.

## Reproducibility

- All CV, HPO, and training entry points accept `--seed` (default `42`) and route through `home_depot_search.utils.reproducibility`.
- Splits are relevance-stratified and evaluated for query/product leakage in notebook `01` before any model is trained.
- Vectorizers and scalers are fit on the training fold only, never on the full dataset.
- Every number quoted in a decision notebook is either computed in the cell directly above it or reported as missing evidence — nothing is hand-typed into the markdown.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `FileNotFoundError` / `Missing required raw input` | Kaggle CSVs not in `data/` | Download the competition data and place the four CSVs under `data/`. |
| `ModuleNotFoundError: No module named 'torch'` during `pytest` collection | Only the base `dev` group is installed | `uv sync --all-groups`, or run `uv run pytest --ignore=tests/models/test_deberta_submission.py --ignore=tests/models/test_transformer_trainer.py` for a torch-free subset. |
| `tests/models/test_hpo.py::TestHPO::test_sample_params_ridge` / `test_build_hpo_pipeline` fail | Known bug: `hpo.py` calls a nonexistent `Trial.log_float`; the pipeline test also feeds a corpus smaller than its configured SVD components | Pre-existing, unrelated to a fresh checkout; not yet fixed. |
| Notebook 01 reports `needs-new-evidence: local DeBERTa tokenizer is unavailable` | `sentencepiece` is not installed | Add `sentencepiece` to the optional group and re-run, or accept the diagnostic — the notebook is designed to report this rather than silently skip tokenizer evidence. |
| `uv run jupytext --sync` reports the `.ipynb` and `.py` disagree | One side was hand-edited without re-syncing | Edit the `.py` source, then run `uv run jupytext --sync notebooks/<name>.ipynb` before re-executing. |

## Development

```bash
git clone <this-repo>
cd Kaggle_Home_Depot_Search
uv sync --all-groups
uv run pytest
uv run ruff check .
```

Notebook edit loop: change the `.py` source → `uv run jupytext --sync notebooks/<name>.ipynb` → `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb` → `uv run pytest tests/notebooks/`.

## License

[MIT](LICENSE)
