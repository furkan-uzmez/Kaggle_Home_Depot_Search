import pytest

from home_depot_search.config import Config, load_config


def test_load_valid_config(tmp_path):
    config_content = """
data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  attributes_path: "data/attributes.csv"
  product_descriptions_path: "data/product_descriptions.csv"

model:
  name: "test_model"
  random_state: 123
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)

    config = load_config(str(config_file))

    assert isinstance(config, Config)
    assert config.data.train_path == "data/train.csv"
    assert config.model.name == "test_model"
    assert config.model.random_state == 123


def test_load_missing_file():
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config("nonexistent_config.yaml")


def test_load_invalid_yaml(tmp_path):
    config_content = """
data:
  train_path: "data/train.csv
  test_path: "data/test.csv"
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)

    with pytest.raises(ValueError, match="Failed to parse YAML configuration"):
        load_config(str(config_file))


def test_load_invalid_schema(tmp_path):
    config_content = """
data:
  train_path: "data/train.csv"
  # Missing other required fields
model:
  name: "test_model"
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)

    with pytest.raises(ValueError, match="Configuration validation failed"):
        load_config(str(config_file))


def test_load_not_a_dict(tmp_path):
    config_content = """
- item1
- item2
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)

    with pytest.raises(ValueError, match="Configuration must be a YAML dictionary"):
        load_config(str(config_file))
