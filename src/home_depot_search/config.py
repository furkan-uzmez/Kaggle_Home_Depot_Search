from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    train_path: str
    test_path: str
    attributes_path: str
    product_descriptions_path: str


class ModelConfig(BaseModel):
    name: str
    random_state: int = Field(default=42)


class Config(BaseModel):
    data: DataConfig
    model: ModelConfig


def load_config(config_path: str = "configs/default.yaml") -> Config:
    """
    Load and validate configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config object containing validated configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is not valid YAML or fails schema validation.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}")

    if not isinstance(config_dict, dict):
        raise ValueError(
            f"Configuration must be a YAML dictionary, got {type(config_dict).__name__}"
        )

    try:
        return Config(**config_dict)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")
