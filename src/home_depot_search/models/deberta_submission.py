import re
import unicodedata
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 256


def clean_transformer_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value)).lower().replace("\u00a0", " ")
    text = re.sub(r"(\d+)\s*['’]\s*", r"\1 ft ", text)
    text = re.sub(r'(\d+)\s*\"', r"\1 in ", text)
    text = re.sub(r"[^0-9a-zçğıöşü\s\.\-\+]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pair_text(
    test: pd.DataFrame,
    descriptions: pd.DataFrame,
    attributes: pd.DataFrame,
) -> pd.Series:
    test = test.copy()
    descriptions = descriptions.copy()
    attributes = attributes.copy()

    test["search_term_raw"] = test["search_term"].map(clean_transformer_text)
    test["product_title_raw"] = test["product_title"].map(clean_transformer_text)
    descriptions["product_description"] = descriptions["product_description"].map(
        clean_transformer_text
    )
    test = test.merge(descriptions, on="product_uid", how="left", validate="m:1")
    test["product_description"] = test["product_description"].fillna("")

    attributes = attributes.dropna(subset=["product_uid"])
    attributes["name"] = attributes["name"].fillna("").map(clean_transformer_text)
    attributes["value"] = attributes["value"].fillna("").map(clean_transformer_text)
    attributes["attribute_text_raw"] = attributes["name"] + ": " + attributes["value"]
    attribute_text = attributes.groupby("product_uid")["attribute_text_raw"].agg(
        " | ".join
    )
    attribute_text = attribute_text.reset_index()
    test = test.merge(attribute_text, on="product_uid", how="left", validate="m:1")
    test["attribute_text_raw"] = test["attribute_text_raw"].fillna("")
    product_text = (
        test["product_title_raw"]
        + " "
        + test["product_description"]
        + " "
        + test["attribute_text_raw"]
    ).str.strip()
    return (test["search_term_raw"] + " [SEP] " + product_text).str.strip()


class TransformerRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            use_safetensors=True,
        )
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0].to(self.regressor.weight.dtype)
        return self.regressor(self.dropout(pooled)).squeeze(-1)


def generate_submission(
    data_dir: Path,
    weights_path: Path,
    output_path: Path,
    batch_size: int = 64,
) -> pd.DataFrame:
    test = pd.read_csv(data_dir / "test.csv", encoding="ISO-8859-1")
    descriptions = pd.read_csv(
        data_dir / "product_descriptions.csv", encoding="ISO-8859-1"
    )
    attributes = pd.read_csv(data_dir / "attributes.csv", encoding="ISO-8859-1")
    texts = build_pair_text(test, descriptions, attributes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TransformerRegressor().to(device).float()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    predictions = []
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc="DeBERTa GPU inference",
            unit="batch",
        ):
            batch_texts = texts.iloc[start : start + batch_size].tolist()
            batch = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            batch = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
            }
            predictions.extend(model(**batch).float().cpu().numpy())

    submission = pd.DataFrame(
        {
            "id": test["id"].values,
            "relevance": torch.clamp(
                torch.tensor(predictions, dtype=torch.float32), 1.0, 3.0
            ).numpy(),
        }
    )
    if len(submission) != len(test) or submission["relevance"].isna().any():
        raise ValueError("Invalid transformer submission")
    temporary_path = output_path.with_suffix(".tmp")
    submission.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    return submission
