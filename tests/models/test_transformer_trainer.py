import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
transformers = pytest.importorskip("transformers", reason="transformers not installed")

from home_depot_search.models.transformer_trainer import (
    RelevanceDataset,
    TransformerRegressor,
    compute_max_length_from_data,
)


@pytest.fixture
def sample_texts():
    return ["hello world", "a longer text with more words for testing", "short"]


class TestRelevanceDataset:
    def test_len(self, sample_texts):
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        dataset = RelevanceDataset(sample_texts, [1.0, 2.0, 3.0], tokenizer, 10)
        assert len(dataset) == 3

    def test_getitem_returns_dict(self, sample_texts):
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        dataset = RelevanceDataset(sample_texts, [1.0, 2.0, 3.0], tokenizer, 32)
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item


class TestTransformerRegressor:
    def test_model_creation(self):
        model = TransformerRegressor("google/bert_uncased_L-2_H-128_A-2", dropout=0.1)
        assert model.regressor.out_features == 1

    def test_forward_pass(self):
        model = TransformerRegressor("google/bert_uncased_L-2_H-128_A-2", dropout=0.0)
        model.eval()
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        assert outputs["logits"].shape == (2,)
        assert outputs["loss"] is None

    def test_forward_with_labels(self):
        model = TransformerRegressor("google/bert_uncased_L-2_H-128_A-2", dropout=0.0)
        model.train()
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16)
        labels = torch.tensor([1.0, 2.0])
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert outputs["loss"] is not None
        assert outputs["loss"].item() > 0


class TestComputeMaxLength:
    def test_compute_max_length(self, sample_texts):
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        result = compute_max_length_from_data(sample_texts, tokenizer, percentile=0.99)
        assert result > 0
