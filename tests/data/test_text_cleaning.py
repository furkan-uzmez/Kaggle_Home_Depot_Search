import numpy as np
import pandas as pd
import pytest

from home_depot_search.data.text_cleaning import clean_text


class TestCleanText:
    """Tests for the deterministic text cleanup function."""

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            # Basic whitespace normalization
            ("  hello   world  ", "hello world"),
            ("hello\nworld", "hello world"),
            ("hello\tworld", "hello world"),
            ("hello\r\nworld", "hello world"),
            # Unicode normalization (NFKD)
            ("café", "café"),  # The second one is 'cafe' + combining acute accent
            # Null to empty string
            (None, ""),
            (np.nan, ""),
            (float("nan"), ""),
            # Non-string types
            (123, "123"),
            (12.34, "12.34"),
            # Complex combinations
            ("  café \n \t 123  ", "café 123"),
        ],
    )
    def test_clean_text_various_inputs(self, input_text, expected):
        """Test clean_text handles various inputs correctly."""
        assert clean_text(input_text) == expected

    def test_clean_text_on_pandas_series(self):
        """Test clean_text can be applied to a pandas Series."""
        s = pd.Series(["  hello   ", None, np.nan, "café"])
        cleaned = s.apply(clean_text)

        assert cleaned[0] == "hello"
        assert cleaned[1] == ""
        assert cleaned[2] == ""
        # The character lengths or precise matches
        assert "cafe" in cleaned[3]

    def test_clean_text_handles_pd_na(self):
        """Test clean_text handles pd.NA correctly."""
        assert clean_text(pd.NA) == ""
