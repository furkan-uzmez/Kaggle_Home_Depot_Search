#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from home_depot_search.data.data_loader import build_merged_dataset
from home_depot_search.evaluation.splits import make_relevance_stratified_folds
from home_depot_search.models.cross_validation import run_cv_full, run_final_evaluation
from home_depot_search.models.registry import (
    FEATURE_REGISTRY,
    MODEL_REGISTRY,
    get_feature_fn,
    get_model_fn,
    list_features,
    list_models,
)
from home_depot_search.utils.logging import RunLogger, init_experiment_csv


def cmd_eda(train_df, test_df, args):
    from home_depot_search.analysis.eda import run_eda, sequence_length_report
    eda_path = run_eda(train_df, test_df, output_dir=args.report_dir)
    print(f"EDA report saved to: {eda_path}")
    seq_path = sequence_length_report(train_df, output_dir=args.report_dir)
    print(f"Sequence length report saved to: {seq_path}")


def cmd_hpo(train_df, args):
    from home_depot_search.models.hpo import objective_factory
    import optuna

    print(f"Running HPO with {args.hpo_trials} trials, {args.hpo_folds} folds...")
    study = optuna.create_study(direction="minimize", study_name="hpo_ridge_tfidf")
    study.optimize(
        objective_factory(
            train_df["product_text_raw"],
            train_df["relevance"],
            lambda trial, model_type, feature_type: {
                "alpha": trial.log_float("alpha", 1e-3, 1e2, log=True),
                "solver": trial.suggest_categorical("solver", ["svd", "lsqr", "saga"]),
                "max_features": trial.suggest_int("max_features", 1000, 20000, log=True),
                "ngram_range_max": trial.suggest_int("ngram_range_max", 1, 3),
                "svd_components": trial.suggest_int("svd_components", 50, 500, log=True),
            },
            n_folds=args.hpo_folds,
            seed=args.seed,
        ),
        n_trials=args.hpo_trials,
    )
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    out_dir = Path(args.log_dir) / "hpo"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "hpo_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
        }, f, indent=2)
    print(f"HPO results saved to: {results_path}")


def cmd_error_analysis(train_df, oof_df, args):
    from home_depot_search.analysis.error_analysis import run_error_analysis

    y_true = train_df["relevance"].values
    matched = train_df[["id", "relevance"]].merge(
        oof_df[["id", "prediction"]], on="id", how="inner"
    )
    if len(matched) == 0:
        print("Error: no matching IDs between training data and predictions.", file=sys.stderr)
        sys.exit(1)
    report_path = run_error_analysis(
        matched["relevance"].values,
        matched["prediction"].values,
        train_df.loc[matched.index],
        output_dir=args.report_dir,
    )
    print(f"Error analysis report saved to: {report_path}")


def cmd_transformer_train(train_df, args):
    from home_depot_search.models.transformer_trainer import (
        TransformerRegressor,
        prepare_loaders,
        train_transformer_epoch,
        evaluate_transformer,
        compute_max_length_from_data,
    )
    from transformers import AutoTokenizer
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = args.transformer_model
    max_length = args.transformer_max_length
    batch_size = args.transformer_batch_size
    epochs = args.transformer_epochs
    lr = args.transformer_lr

    texts = train_df["product_text_raw"].fillna("")
    targets = train_df["relevance"].values

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if max_length <= 0:
        max_length = compute_max_length_from_data(texts, tokenizer, percentile=0.99)
        max_length = min(max_length, 512)

    split = int(len(train_df) * 0.8)
    train_texts, val_texts = texts[:split], texts[split:]
    train_targets, val_targets = targets[:split], targets[split:]

    train_loader, val_loader = prepare_loaders(
        train_texts, train_targets, val_texts, val_targets,
        tokenizer, max_length, batch_size,
    )

    model = TransformerRegressor(model_name, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    print(f"Training {model_name} | max_len={max_length} | batch={batch_size} | epochs={epochs} | lr={lr} | {len(train_loader)} steps/epoch")
    for epoch in range(epochs):
        train_loss = train_transformer_epoch(model, train_loader, optimizer, scheduler, device)
        val_rmse, val_loss = evaluate_transformer(model, val_loader, device)
        print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_rmse={val_rmse:.4f}")

    val_rmse, val_loss = evaluate_transformer(model, val_loader, device)
    print(f"Final: val_rmse={val_rmse:.4f} val_loss={val_loss:.4f}")

    out_dir = Path(args.output_dir) / "transformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_name = model_name.replace("/", "_")
    torch.save(model.state_dict(), str(out_dir / f"{save_name}.pt"))
    print(f"Model saved to: {out_dir / save_name}.pt")


def cmd_submission(train_df, test_df, oof_df, args):
    feature_name = args.submission_feature or "tfidf-svd"
    model_name = args.submission_model or "ridge"

    feature_fn = get_feature_fn(feature_name, seed=args.seed)

    y_train = train_df["relevance"].values
    X_train = feature_fn(train_df)
    X_test = feature_fn(test_df)

    model = get_model_fn(model_name, seed=args.seed)
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    test_preds = np.clip(test_preds, 1.0, 3.0)

    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "relevance": test_preds,
    })
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to: {sub_path}  (n={len(submission)})")
    print(f"  Pred range: [{test_preds.min():.3f}, {test_preds.max():.3f}]")
    print(f"  Pred mean:  {test_preds.mean():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Home Depot Search Relevance - Unified Experiment Runner"
    )
    parser.add_argument(
        "--features", "-f", nargs="+",
        help="Feature names (default: all from registry)",
    )
    parser.add_argument(
        "--models", "-m", nargs="+",
        help="Model names (default: all from registry)",
    )
    parser.add_argument(
        "--n-folds", "-k", type=int, default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--log-dir", default="logs",
        help="Log directory (default: logs)",
    )
    parser.add_argument(
        "--output-dir", default="artifacts",
        help="Output directory for artifacts (default: artifacts)",
    )
    parser.add_argument(
        "--report-dir", default="outputs/reports",
        help="Output directory for reports (default: outputs/reports)",
    )
    parser.add_argument(
        "--list-features", action="store_true",
        help="List available features and exit",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available models and exit",
    )

    # EDA
    parser.add_argument(
        "--run-eda", action="store_true",
        help="Run EDA and sequence length analysis, then exit",
    )

    # HPO
    parser.add_argument(
        "--hpo", action="store_true",
        help="Run hyperparameter optimization with Optuna, then exit",
    )
    parser.add_argument(
        "--hpo-trials", type=int, default=50,
        help="Number of HPO trials (default: 50)",
    )
    parser.add_argument(
        "--hpo-folds", type=int, default=3,
        help="Number of HPO CV folds (default: 3)",
    )

    # Error analysis
    parser.add_argument(
        "--run-error-analysis", action="store_true",
        help="Run error analysis on CV predictions, then exit",
    )
    parser.add_argument(
        "--error-predictions-csv", type=str,
        help="Path to OOF predictions CSV (required for --run-error-analysis without prior CV)",
    )

    # Transformer training
    parser.add_argument(
        "--transformer-train", action="store_true",
        help="Train a transformer model on the full dataset",
    )
    parser.add_argument(
        "--transformer-model", type=str, default="microsoft/deberta-v3-small",
        help="Transformer model name (default: microsoft/deberta-v3-small)",
    )
    parser.add_argument(
        "--transformer-max-length", type=int, default=256,
        help="Max sequence length (0 = auto from data, default: 256)",
    )
    parser.add_argument(
        "--transformer-batch-size", type=int, default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--transformer-epochs", type=int, default=3,
        help="Number of epochs (default: 3)",
    )
    parser.add_argument(
        "--transformer-lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)",
    )

    # Submission
    parser.add_argument(
        "--make-submission", action="store_true",
        help="Train on full data and generate Kaggle submission CSV",
    )
    parser.add_argument(
        "--submission-feature", type=str, default="tfidf-svd",
        help="Feature set for submission (default: tfidf-svd)",
    )
    parser.add_argument(
        "--submission-model", type=str, default="ridge",
        help="Model for submission (default: ridge)",
    )

    args = parser.parse_args()

    if args.list_features:
        print("Available features:")
        for name, desc in list_features():
            print(f"  {name:30s} {desc}")
        return

    if args.list_models:
        print("Available models:")
        for name, class_name in list_models():
            print(f"  {name:30s} {class_name}")
        return

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    desc_path = data_dir / "product_descriptions.csv"
    attr_path = data_dir / "attributes.csv"

    try:
        train_df, test_df = build_merged_dataset(
            str(train_path), str(test_path), str(desc_path), str(attr_path),
        )
    except FileNotFoundError:
        print(
            f"Error: Data files not found in {data_dir}/.\n"
            f"Please download the Kaggle dataset and place the CSV files in {data_dir}/",
            file=sys.stderr,
        )
        sys.exit(1)

    # EDA-only mode
    if args.run_eda:
        cmd_eda(train_df, test_df, args)
        return

    # HPO-only mode
    if args.hpo:
        cmd_hpo(train_df, args)
        return

    # Transformer-only mode
    if args.transformer_train:
        cmd_transformer_train(train_df, args)
        return

    # --- Full CV experiment ---
    feature_names = args.features or list(FEATURE_REGISTRY.keys())
    model_names = args.models or list(MODEL_REGISTRY.keys())

    for name in feature_names:
        if name not in FEATURE_REGISTRY:
            print(
                f"Error: Unknown feature '{name}'. Use --list-features to see available features.",
                file=sys.stderr,
            )
            sys.exit(1)

    for name in model_names:
        if name not in MODEL_REGISTRY:
            print(
                f"Error: Unknown model '{name}'. Use --list-models to see available models.",
                file=sys.stderr,
            )
            sys.exit(1)

    fold_df = make_relevance_stratified_folds(
        train_df, n_splits=args.n_folds, seed=args.seed,
    )

    csv_path = init_experiment_csv(args.log_dir)
    results = []

    for feature_name in feature_names:
        for model_name in model_names:
            run_id = f"{feature_name}_{model_name}_{args.seed}"

            logger = RunLogger(run_id, log_dir=args.log_dir)
            logger.log_metadata("model_name", model_name)
            logger.log_metadata("feature_set", feature_name)
            logger.log_metadata("seed", args.seed)

            feature_fn = get_feature_fn(feature_name, seed=args.seed)

            def make_model_fn(model_name, seed):
                def _model_fn(X, y):
                    model = get_model_fn(model_name, seed=seed)
                    model.fit(X, y)
                    return model
                return _model_fn

            model_fn = make_model_fn(model_name, seed=args.seed)

            oof_df, cv_metrics = run_cv_full(
                train_df, fold_df, feature_fn, model_fn, seed=args.seed,
            )

            final_metrics = run_final_evaluation(oof_df)

            logger.log_metadata("mean_rmse", final_metrics["mean_rmse"])
            logger.log_metadata("std_rmse", final_metrics["std_rmse"])
            logger.log_metadata("mean_clipped_rmse", final_metrics["mean_clipped_rmse"])

            oof_path = Path(args.log_dir) / "runs" / f"{run_id}_oof.csv"
            oof_df.to_csv(oof_path, index=False)
            logger.log_metadata("oof_path", str(oof_path))

            artifact_dir = Path(args.output_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / f"{run_id}.json"
            with open(artifact_path, "w") as f:
                json.dump(final_metrics, f, indent=2)
            logger.log_metadata("artifact_path", str(artifact_path))

            logger.save_run_json()
            logger.append_to_experiment_csv(csv_path)

            results.append({
                "feature": feature_name,
                "model": model_name,
                "mean_rmse": final_metrics["mean_rmse"],
                "std_rmse": final_metrics["std_rmse"],
                "mean_clipped_rmse": final_metrics["mean_clipped_rmse"],
            })

    print("\nResults:")
    header = (
        f"  {'Feature':22s} {'Model':18s} {'Mean RMSE':12s} {'Std RMSE':12s} "
        f"{'Mean Clipped RMSE':18s}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))
    for r in results:
        print(
            f"  {r['feature']:22s} {r['model']:18s} "
            f"{r['mean_rmse']:.4f}      {r['std_rmse']:.4f}      "
            f"{r['mean_clipped_rmse']:.4f}"
        )

    # Error analysis uses the last run's OOF
    if args.run_error_analysis and results:
        last_oof_path = Path(args.log_dir) / "runs" / f"{results[-1]['feature']}_{results[-1]['model']}_{args.seed}_oof.csv"
        if last_oof_path.exists():
            oof_df = pd.read_csv(last_oof_path)
            cmd_error_analysis(train_df, oof_df, args)

    # Submission
    if args.make_submission:
        last_oof_path = Path(args.log_dir) / "runs" / f"{results[-1]['feature']}_{results[-1]['model']}_{args.seed}_oof.csv"
        if last_oof_path.exists():
            oof_df = pd.read_csv(last_oof_path)
        else:
            oof_df = pd.DataFrame()
        cmd_submission(train_df, test_df, oof_df, args)


if __name__ == "__main__":
    main()
