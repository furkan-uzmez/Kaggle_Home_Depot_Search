import pandas as pd
import pytest

from home_depot_search.analysis.eda import (
    compute_length_stats,
    compute_token_stats,
    compute_relevance_stats,
    compute_top_tokens,
    compute_token_sequence_stats,
    run_eda,
    sequence_length_report,
    compute_max_length,
)
from home_depot_search.models.padding import PaddingCollator, compute_sequence_lengths


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "search_term_raw": ["hello world", "foo bar baz", "a"],
        "product_title_raw": ["big product", "another thing here", "short"],
        "product_description": ["description one", "desc two with more words", "d"],
        "attribute_text_raw": ["attr1 attr2", "attr3", "attr4 attr5 attr6"],
        "product_text_raw": [
            "big product description one attr1 attr2",
            "another thing here desc two with more words attr3",
            "short d attr4 attr5 attr6",
        ],
        "relevance": [1.0, 2.0, 3.0],
    })


class TestLengthStats:
    def test_compute_length_stats(self, sample_df):
        stats = compute_length_stats(sample_df["search_term_raw"], "search_term_raw")
        assert stats["field"] == "search_term_raw"
        assert stats["count"] == 3
        assert stats["null_count"] == 0
        assert stats["mean_length"] == pytest.approx(7.67, abs=0.1)

    def test_compute_token_stats(self, sample_df):
        stats = compute_token_stats(sample_df["search_term_raw"], "search_term_raw")
        assert stats["field"] == "search_term_raw"
        assert stats["mean_tokens"] == pytest.approx(2.0, abs=0.1)

    def test_compute_token_sequence_stats(self, sample_df):
        stats = compute_token_sequence_stats(sample_df["product_text_raw"], "product_text_raw")
        assert stats["mean_tokens"] > 0
        assert stats["p50_tokens"] <= stats["p99_tokens"]

    def test_compute_relevance_stats(self, sample_df):
        stats = compute_relevance_stats(sample_df)
        assert stats["count"] == 3
        assert stats["mean"] == 2.0
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert len(stats["histogram"]) == 10

    def test_compute_top_tokens(self, sample_df):
        tokens = compute_top_tokens(sample_df["search_term_raw"], "search_term_raw", top_n=5)
        assert "top_tokens" in tokens
        assert len(tokens["top_tokens"]) == 5


class TestEDAReport:
    def test_run_eda(self, sample_df, tmp_path):
        report_path = run_eda(sample_df, sample_df, output_dir=str(tmp_path))
        assert tmp_path.joinpath("eda_report.md").exists()
        text = tmp_path.joinpath("eda_report.md").read_text()
        assert "EDA Report" in text
        assert "search_term_raw" in text

    def test_sequence_length_report(self, sample_df, tmp_path):
        report_path = sequence_length_report(sample_df, output_dir=str(tmp_path))
        assert tmp_path.joinpath("sequence_length_report.md").exists()
        text = tmp_path.joinpath("sequence_length_report.md").read_text()
        assert "Sequence Length" in text

    def test_compute_max_length(self, sample_df):
        ml = compute_max_length(sample_df, "product_text_raw", 0.95)
        assert ml > 0
        assert isinstance(ml, int)


class TestPadding:
    def test_padding_collator(self):
        collator = PaddingCollator(max_length=10)
        result = collator([[1, 2, 3], [4, 5, 6, 7, 8]])
        assert result["input_ids"].shape == (2, 10)
        assert result["attention_mask"][0, :3].sum() == 3
        assert result["attention_mask"][0, 3:].sum() == 0

    def test_padding_truncation(self):
        collator = PaddingCollator(max_length=3)
        result = collator([[1, 2, 3, 4, 5], [6, 7]])
        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"][0].sum() == 3

    def test_compute_sequence_lengths(self, sample_df):
        lengths = compute_sequence_lengths(sample_df["search_term_raw"])
        assert len(lengths) == 3
        assert lengths.tolist() == [2, 3, 1]
