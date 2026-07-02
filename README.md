# Home Depot Search Relevance

Reproducible Python pipeline for the Kaggle Home Depot Product Search Relevance task. The project builds merged product/search text features, runs leakage-aware cross-validation, logs experiment metrics, and can generate a Kaggle submission file.

## Quick Start

```bash
uv sync
uv run pytest
uv run python run_experiments.py --list-features
uv run python run_experiments.py --features tfidf-svd text-overlap-v2 --models ridge
```

## Data

Place the Kaggle CSV files under `data/`:

- `train.csv`
- `test.csv`
- `attributes.csv`
- `product_descriptions.csv`

The `data/` directory is ignored by Git and should not be committed.

## Useful Commands

```bash
uv run python run_experiments.py --run-eda
uv run python run_experiments.py --hpo --hpo-trials 20 --hpo-folds 3
uv run python run_experiments.py --make-submission --submission-feature tfidf-svd --submission-model ridge
```

Generated experiment outputs are written under `artifacts/`, `logs/`, and `outputs/`; these are ignored by Git.
