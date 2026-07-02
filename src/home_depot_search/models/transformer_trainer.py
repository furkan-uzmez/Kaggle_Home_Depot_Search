import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error

from home_depot_search.utils.reproducibility import set_reproducibility
from home_depot_search.models.padding import compute_sequence_lengths


class RelevanceDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length):
        self.targets = torch.tensor(list(targets), dtype=torch.float) if targets is not None else None
        tokenized = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.targets is not None:
            item["labels"] = self.targets[idx]
        return item


class TransformerRegressor(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.regressor(pooled).squeeze(-1)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


def train_transformer_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_transformer(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs["loss"].item()
            all_preds.extend(outputs["logits"].cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    rmse = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
    return rmse, total_loss / len(loader)


def compute_max_length_from_data(texts, tokenizer, percentile=0.99):
    tokenizer_len_fn = lambda x: len(tokenizer.encode(x, truncation=True))
    lengths = compute_sequence_lengths(texts, tokenizer_len_fn=tokenizer_len_fn)
    p99 = int(np.percentile(lengths, percentile))
    return min(p99, 512)


def prepare_loaders(
    train_texts,
    train_targets,
    val_texts,
    val_targets,
    tokenizer,
    max_length,
    batch_size,
):
    train_dataset = RelevanceDataset(train_texts, train_targets, tokenizer, max_length)
    val_dataset = RelevanceDataset(val_texts, val_targets, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
