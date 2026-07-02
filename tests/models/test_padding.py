import numpy as np
import pytest

from home_depot_search.models.padding import (
    PaddingCollator,
    truncate_text,
    compute_sequence_lengths,
)


class TestPaddingCollator:
    def test_padding(self):
        collator = PaddingCollator(max_length=10, pad_token_id=0)
        result = collator([[1, 2, 3], [4, 5, 6, 7]])
        assert result["input_ids"].shape == (2, 10)
        assert result["input_ids"][0, 3:].sum() == 0
        assert result["attention_mask"][0, :3].sum() == 3
        assert result["attention_mask"][0, 3:].sum() == 0

    def test_truncation(self):
        collator = PaddingCollator(max_length=3)
        result = collator([[1, 2, 3, 4, 5, 6]])
        assert result["input_ids"].shape == (1, 3)
        assert list(result["input_ids"][0]) == [1, 2, 3]

    def test_overflow_protection(self):
        collator = PaddingCollator(max_length=2)
        result = collator([[1], [2, 3, 4, 5]])
        assert result["input_ids"].shape == (2, 2)

    def test_single_element(self):
        collator = PaddingCollator(max_length=5)
        result = collator([[42]])
        assert result["input_ids"][0, 0] == 42
        assert result["attention_mask"][0, 0] == 1
        assert result["attention_mask"][0, 1:].sum() == 0


class TestTruncateText:
    def test_short_text(self):
        result = truncate_text("hello world", 10, lambda x: len(x.split()))
        assert result == "hello world"

    def test_long_text(self):
        text = "a b c d e f g h i j k l m n o p"
        result = truncate_text(text, 3, lambda x: len(x.split()))
        tokens = result.split()
        assert len(tokens) <= 10


class TestComputeSequenceLengths:
    def test_whitespace_split(self):
        import pandas as pd
        s = pd.Series(["hello world", "foo bar baz", "short"])
        lengths = compute_sequence_lengths(s)
        assert lengths.tolist() == [2, 3, 1]
