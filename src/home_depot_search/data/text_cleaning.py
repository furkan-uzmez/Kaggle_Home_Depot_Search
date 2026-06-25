import re
import unicodedata
from typing import Any

import pandas as pd


def clean_text(text: Any) -> str:
    """
    Apply deterministic, lossless text cleanup to raw data.

    This function normalizes Unicode (NFKD), replaces all whitespace
    (including newlines and tabs) with a single space, strips leading/trailing
    whitespace, and converts nulls/NaNs to empty strings.

    It does NOT apply target-aware changes like stemming, lemmatization,
    stopword removal, or lowercasing.

    Args:
        text: The input text or value to clean.

    Returns:
        The cleaned string.
    """
    if pd.isna(text) or text is None:
        return ""

    # Convert to string if it's not already
    text_str = str(text)

    # Normalize Unicode (NFKD)
    # NFKD decomposes characters, e.g., 'é' becomes 'e' + combining acute accent
    normalized_text = unicodedata.normalize("NFKD", text_str)

    # Normalize whitespace: replace multiple spaces, tabs, newlines with a single space
    # and strip leading/trailing whitespace
    whitespace_normalized = re.sub(r"\s+", " ", normalized_text).strip()

    return whitespace_normalized
