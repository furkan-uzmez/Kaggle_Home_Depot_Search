from pathlib import Path

import numpy as np
import pandas as pd


def run_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    output_dir: str = "outputs/reports",
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)

    lines = []
    lines.append("# Error Analysis Report\n")
    lines.append(f"Samples: {len(y_true):,}\n")

    lines.append("## Global Error Metrics\n")
    lines.append(f"- Mean Absolute Error (MAE): {float(np.mean(abs_errors)):.4f}")
    lines.append(f"- Root Mean Squared Error (RMSE): {float(np.sqrt(np.mean(residuals**2))):.4f}")
    lines.append(f"- Mean Error (bias): {float(np.mean(residuals)):.4f}")
    lines.append(f"- Std of Residuals: {float(np.std(residuals)):.4f}")

    lines.append("\n## Error Distribution (bucketed)\n")
    bins = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (abs_errors >= lo) & (abs_errors < hi)
        count = int(mask.sum())
        pct = float(mask.mean() * 100)
        bar = "#" * int(pct / 2)
        lines.append(f"  [{lo:.2f}, {hi:.2f}): {count:>6} ({pct:5.1f}%) {bar}")

    if "search_term_raw" in df.columns:
        df_err = df.copy()
        df_err["abs_error"] = abs_errors
        df_err["residual"] = residuals
        df_err["predicted"] = y_pred
        df_err["actual"] = y_true

        lines.append("\n## Error by Search Term Length\n")
        df_err["search_len"] = df_err["search_term_raw"].str.len()
        bins_len = [0, 10, 20, 30, 50, 100, 200]
        for i in range(len(bins_len) - 1):
            lo, hi = bins_len[i], bins_len[i + 1]
            mask = (df_err["search_len"] >= lo) & (df_err["search_len"] < hi)
            if mask.sum() == 0:
                continue
            lines.append(
                f"  [{lo:>3}, {hi:>3}): n={int(mask.sum()):>5}, "
                f"MAE={df_err.loc[mask, 'abs_error'].mean():.3f}, "
                f"RMSE={np.sqrt((df_err.loc[mask, 'residual']**2).mean()):.3f}"
            )

        lines.append("\n## Worst Predictions (top 20 by abs error)\n")
        worst = df_err.nlargest(20, "abs_error")
        lines.append("| # | Search Term | Actual | Predicted | Abs Error |")
        lines.append("|---|-------------|-------:|----------:|----------:|")
        for i, (_, row) in enumerate(worst.iterrows()):
            search = str(row.get("search_term_raw", ""))[:60]
            lines.append(
                f"| {i+1} | {search} | {row['actual']:.2f} | {row['predicted']:.2f} | {row['abs_error']:.3f} |"
            )

    lines.append("\n## Prediction vs Actual\n")
    for bucket in [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]:
        lo, hi = bucket
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            continue
        lines.append(
            f"  Actual [{lo:.1f}, {hi:.1f}): n={int(mask.sum()):>5}, "
            f"pred_mean={float(np.mean(y_pred[mask])):.3f}, "
            f"abs_error={float(np.mean(abs_errors[mask])):.3f}"
        )

    report = "\n".join(lines)
    report_path = output_path / "error_analysis_report.md"
    report_path.write_text(report, encoding="utf-8")
    return str(report_path)
