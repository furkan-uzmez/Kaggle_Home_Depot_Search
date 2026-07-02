from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


TEXT_FIELDS = [
    "search_term_raw",
    "product_title_raw",
    "product_description",
    "attribute_text_raw",
    "product_text_raw",
]


def compute_length_stats(series: pd.Series, name: str) -> dict:
    lengths = series.str.len()
    return {
        "field": name,
        "count": int(len(series)),
        "null_count": int(series.isna().sum()),
        "empty_count": int((lengths == 0).sum()),
        "min_length": int(lengths.min()) if len(lengths) > 0 else 0,
        "mean_length": float(lengths.mean()) if len(lengths) > 0 else 0.0,
        "std_length": float(lengths.std()) if len(lengths) > 0 else 0.0,
        "max_length": int(lengths.max()) if len(lengths) > 0 else 0,
        "p25_length": float(lengths.quantile(0.25)) if len(lengths) > 0 else 0.0,
        "p50_length": float(lengths.quantile(0.50)) if len(lengths) > 0 else 0.0,
        "p75_length": float(lengths.quantile(0.75)) if len(lengths) > 0 else 0.0,
        "p90_length": float(lengths.quantile(0.90)) if len(lengths) > 0 else 0.0,
        "p99_length": float(lengths.quantile(0.99)) if len(lengths) > 0 else 0.0,
    }


def compute_token_stats(series: pd.Series, name: str) -> dict:
    token_counts = series.str.split().str.len()
    return {
        "field": name,
        "mean_tokens": float(token_counts.mean()) if len(token_counts) > 0 else 0.0,
        "std_tokens": float(token_counts.std()) if len(token_counts) > 0 else 0.0,
        "median_tokens": float(token_counts.median()) if len(token_counts) > 0 else 0.0,
        "p90_tokens": float(token_counts.quantile(0.90)) if len(token_counts) > 0 else 0.0,
        "p99_tokens": float(token_counts.quantile(0.99)) if len(token_counts) > 0 else 0.0,
        "max_tokens": int(token_counts.max()) if len(token_counts) > 0 else 0,
    }


def compute_token_sequence_stats(series: pd.Series, name: str, tokenizer_len_fn=None) -> dict:
    if tokenizer_len_fn is not None:
        token_lens = series.apply(lambda x: tokenizer_len_fn(x) if isinstance(x, str) else 0)
    else:
        token_lens = series.str.split().str.len()
    return {
        "field": name,
        "mean_tokens": float(token_lens.mean()) if len(token_lens) > 0 else 0.0,
        "std_tokens": float(token_lens.std()) if len(token_lens) > 0 else 0.0,
        "p50_tokens": float(token_lens.quantile(0.50)) if len(token_lens) > 0 else 0.0,
        "p90_tokens": float(token_lens.quantile(0.90)) if len(token_lens) > 0 else 0.0,
        "p95_tokens": float(token_lens.quantile(0.95)) if len(token_lens) > 0 else 0.0,
        "p99_tokens": float(token_lens.quantile(0.99)) if len(token_lens) > 0 else 0.0,
        "max_tokens": int(token_lens.max()) if len(token_lens) > 0 else 0,
    }


def compute_relevance_stats(df: pd.DataFrame) -> dict:
    rel = df["relevance"]
    return {
        "count": int(len(rel)),
        "mean": float(rel.mean()),
        "std": float(rel.std()),
        "min": float(rel.min()),
        "max": float(rel.max()),
        "p25": float(rel.quantile(0.25)),
        "p50": float(rel.quantile(0.50)),
        "p75": float(rel.quantile(0.75)),
        "histogram": [
            {"bin": float(b), "count": int(c)}
            for b, c in zip(
                np.histogram(rel, bins=10, range=(1.0, 3.0))[1].tolist(),
                np.histogram(rel, bins=10, range=(1.0, 3.0))[0].tolist(),
            )
        ],
    }


def compute_top_tokens(series: pd.Series, name: str, top_n: int = 50) -> dict:
    all_tokens = " ".join(series.dropna().astype(str)).lower().split()
    most_common = Counter(all_tokens).most_common(top_n)
    return {
        "field": name,
        "total_tokens": len(all_tokens),
        "unique_tokens": len(set(all_tokens)),
        "top_tokens": [{"token": t, "count": c} for t, c in most_common],
    }


def run_eda(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "outputs/reports",
    top_n_tokens: int = 50,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sections = []

    sections.append("# EDA Report — Home Depot Search Relevance\n")
    sections.append(f"Train rows: {len(train_df):,}, Test rows: {len(test_df):,}\n")

    sections.append("## Relevance Distribution\n")
    rel_stats = compute_relevance_stats(train_df)
    sections.append(f"- Mean: {rel_stats['mean']:.3f} ± {rel_stats['std']:.3f}")
    sections.append(f"- Range: [{rel_stats['min']:.1f}, {rel_stats['max']:.1f}]")
    sections.append(f"- Median (P50): {rel_stats['p50']:.2f}")
    sections.append(f"- Histogram (10 bins 1.0–3.0):")
    for h in rel_stats["histogram"]:
        sections.append(f"  - {h['bin']:.1f}+ : {h['count']}")

    sections.append("\n## Text Field Length Statistics\n")
    sections.append("| Field | Count | Null | Empty | Mean±Std | P50 | P90 | P99 | Max |")
    sections.append("|-------|------:|-----:|------:|---------:|----:|----:|----:|----:|")
    for field in TEXT_FIELDS:
        if field in train_df.columns:
            s = compute_length_stats(train_df[field], field)
            sections.append(
                f"| {s['field']} | {s['count']:,} | {s['null_count']} | {s['empty_count']:,} "
                f"| {s['mean_length']:.1f}±{s['std_length']:.1f} | {s['p50_length']:.0f} "
                f"| {s['p90_length']:.0f} | {s['p99_length']:.0f} | {s['max_length']:,} |"
            )

    sections.append("\n## Text Field Token Statistics (whitespace split)\n")
    sections.append("| Field | Mean±Std Tokens | Median | P90 | P99 | Max |")
    sections.append("|-------|----------------:|-------:|----:|----:|----:|")
    for field in TEXT_FIELDS:
        if field in train_df.columns:
            s = compute_token_stats(train_df[field], field)
            sections.append(
                f"| {s['field']} | {s['mean_tokens']:.1f}±{s['std_tokens']:.1f} "
                f"| {s['median_tokens']:.0f} | {s['p90_tokens']:.0f} "
                f"| {s['p99_tokens']:.0f} | {s['max_tokens']:,} |"
            )

    sections.append("\n## Top Tokens by Field\n")
    for field in TEXT_FIELDS:
        if field in train_df.columns and field in ("search_term_raw", "product_title_raw"):
            t = compute_top_tokens(train_df[field], field, top_n=top_n_tokens)
            sections.append(f"\n### {field} — {t['total_tokens']:,} total, {t['unique_tokens']:,} unique\n")
            sections.append("| # | Token | Count |")
            sections.append("|---|-------|------:|")
            for i, tok in enumerate(t["top_tokens"][:20]):
                sections.append(f"| {i+1} | `{tok['token']}` | {tok['count']:,} |")

    report = "\n".join(sections)
    report_path = output_path / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")

    return str(report_path)


def compute_max_length(
    df: pd.DataFrame,
    text_field: str = "product_text_raw",
    percentile: float = 0.99,
    tokenizer_len_fn=None,
) -> int:
    stats = compute_token_sequence_stats(df[text_field], text_field, tokenizer_len_fn=tokenizer_len_fn)
    if percentile <= 0.95:
        return int(stats["p95_tokens"])
    return int(stats["p99_tokens"])


def sequence_length_report(
    df: pd.DataFrame,
    output_dir: str = "outputs/reports",
    tokenizer_len_fn=None,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Sequence Length Analysis\n")
    lines.append(f"Dataset rows: {len(df):,}\n")

    lines.append("## Sequence Length by Field\n")
    lines.append("| Field | Mean | Std | P50 | P90 | P95 | P99 | Max |")
    lines.append("|-------|-----:|----:|----:|----:|----:|----:|----:|")

    recommendations = []
    for field in TEXT_FIELDS:
        if field in df.columns:
            s = compute_token_sequence_stats(df[field], field, tokenizer_len_fn=tokenizer_len_fn)
            lines.append(
                f"| {s['field']} | {s['mean_tokens']:.1f} | {s['std_tokens']:.1f} "
                f"| {s['p50_tokens']:.0f} | {s['p90_tokens']:.0f} "
                f"| {s['p95_tokens']:.0f} | {s['p99_tokens']:.0f} | {s['max_tokens']} |"
            )
            p99 = int(s["p99_tokens"])
            p95 = int(s["p95_tokens"])
            recommendations.append(
                f"- **{field}**: P95={p95}, P99={p99}, max={s['max_tokens']} → "
                f"recommended max_length = {p99}"
            )

    lines.append("\n## Recommended max_length Values\n")
    lines.append("Based on P99 (only 1% of samples truncated):\n")
    lines.extend(recommendations)

    lines.append("\n### Recommendation\n")
    product_p99 = compute_max_length(df, "product_text_raw", 0.99, tokenizer_len_fn=tokenizer_len_fn)
    product_p95 = compute_max_length(df, "product_text_raw", 0.95, tokenizer_len_fn=tokenizer_len_fn)
    lines.append(
        f"- **Conservative (P99)** for `product_text_raw`: **{product_p99}** tokens\n"
        f"  → 1% of samples will be truncated.\n"
        f"- **Aggressive (P95)** for `product_text_raw`: **{product_p95}** tokens\n"
        f"  → 5% of samples will be truncated.\n"
        f"- **Safe default**: round up to next power of 2: **{2**int(np.ceil(np.log2(product_p99)))}**\n"
    )

    report = "\n".join(lines)
    report_path = output_path / "sequence_length_report.md"
    report_path.write_text(report, encoding="utf-8")

    return str(report_path)
