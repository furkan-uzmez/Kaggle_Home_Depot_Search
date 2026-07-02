import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def get_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_experiment_csv(log_dir: str = "logs") -> Path:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    csv_path = log_path / "experiments.csv"
    if not csv_path.exists():
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "run_id",
                    "timestamp",
                    "model_name",
                    "feature_set",
                    "config_hash",
                    "data_manifest_hash",
                    "fold_manifest_path",
                    "seed",
                    "mean_rmse",
                    "std_rmse",
                    "mean_clipped_rmse",
                    "oof_path",
                    "artifact_path",
                ]
            )
    return csv_path


class RunLogger:
    def __init__(self, run_id: str, log_dir: str = "logs") -> None:
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.metadata: dict = {}
        self.fold_metrics: list[dict] = []

        runs_dir = self.log_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

    def log_metadata(self, key: str, value) -> None:
        self.metadata[key] = value

    def log_fold_metrics(self, fold: int, rmse: float, clipped_rmse: float) -> None:
        self.fold_metrics.append(
            {"fold": fold, "rmse": rmse, "clipped_rmse": clipped_rmse}
        )

    def save_run_json(self) -> None:
        data = {**self.metadata, "fold_metrics": self.fold_metrics}
        path = self.log_dir / "runs" / f"{self.run_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def append_to_experiment_csv(self, csv_path: Path) -> None:
        row = {
            "run_id": self.run_id,
            "timestamp": get_timestamp(),
            "model_name": self.metadata.get("model_name", ""),
            "feature_set": self.metadata.get("feature_set", ""),
            "config_hash": self.metadata.get("config_hash", ""),
            "data_manifest_hash": self.metadata.get("data_manifest_hash", ""),
            "fold_manifest_path": self.metadata.get("fold_manifest_path", ""),
            "seed": self.metadata.get("seed", ""),
            "mean_rmse": self.metadata.get("mean_rmse", ""),
            "std_rmse": self.metadata.get("std_rmse", ""),
            "mean_clipped_rmse": self.metadata.get("mean_clipped_rmse", ""),
            "oof_path": self.metadata.get("oof_path", ""),
            "artifact_path": self.metadata.get("artifact_path", ""),
        }
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
