from typing import Callable, Optional

import numpy as np
import pandas as pd


class PaddingCollator:
    def __init__(self, max_length: int, pad_token_id: int = 0):
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __call__(self, input_ids: list[list[int]]) -> dict[str, np.ndarray]:
        batch_size = len(input_ids)
        padded = np.full((batch_size, self.max_length), self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((batch_size, self.max_length), dtype=np.int64)
        for i, seq in enumerate(input_ids):
            seq_len = min(len(seq), self.max_length)
            padded[i, :seq_len] = seq[:seq_len]
            attention_mask[i, :seq_len] = 1
        return {"input_ids": padded, "attention_mask": attention_mask, "max_length": self.max_length}


def truncate_text(text: str, max_length: int, tokenizer_len_fn: Callable) -> str:
    tokens = text.split()
    if len(tokens) <= max_length:
        return text
    while len(tokens) > max_length and tokenizer_len_fn(" ".join(tokens)) > max_length:
        tokens = tokens[:-1]
    return " ".join(tokens)


def make_text_collator(
    max_length: int,
    pad_token_id: int = 0,
) -> PaddingCollator:
    return PaddingCollator(max_length=max_length, pad_token_id=pad_token_id)


def compute_sequence_lengths(
    texts: pd.Series,
    tokenizer_len_fn: Optional[Callable] = None,
) -> np.ndarray:
    if tokenizer_len_fn is not None:
        return np.array([tokenizer_len_fn(t) for t in texts])
    return np.array(texts.str.split().str.len())
