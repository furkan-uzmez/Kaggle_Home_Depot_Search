import json
from unittest.mock import patch

import pytest

from home_depot_search.cli import main, check_data


def test_check_data_missing_files(tmp_path, capsys):
    # Setup mock config with missing files
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  attributes_path: "data/attributes.csv"
  product_descriptions_path: "data/product_descriptions.csv"
model:
  name: "test"
""")

    with pytest.raises(SystemExit) as exc_info:
        main(["--config", str(config_path), "--check-data"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "missing" in captured.err.lower() or "missing" in captured.out.lower()
    assert "data/train.csv" in captured.err or "data/train.csv" in captured.out


def test_check_data_success(tmp_path, capsys):
    # Setup mock config and valid files
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train_file = data_dir / "train.csv"
    train_file.write_text(
        "id,product_uid,product_title,search_term,relevance\n1,1001,A,B,3.0\n"
    )

    test_file = data_dir / "test.csv"
    test_file.write_text("id,product_uid,product_title,search_term\n2,1002,C,D\n")

    attr_file = data_dir / "attributes.csv"
    attr_file.write_text("product_uid,name,value\n1001,Brand,Home Depot\n")

    desc_file = data_dir / "product_descriptions.csv"
    desc_file.write_text("product_uid,product_description\n1001,A great product\n")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
data:
  train_path: "{train_file}"
  test_path: "{test_file}"
  attributes_path: "{attr_file}"
  product_descriptions_path: "{desc_file}"
model:
  name: "test"
""")

    artifacts_dir = tmp_path / "artifacts"
    
    # We need to create a config object
    class ConfigData:
        def __init__(self):
            self.train_path = str(train_file)
            self.test_path = str(test_file)
            self.attributes_path = str(attr_file)
            self.product_descriptions_path = str(desc_file)
            
    class Config:
        def __init__(self):
            self.data = ConfigData()
            
    config = Config()

    check_data(config, artifacts_dir)

    # Verify manifest generated
    manifest_path = artifacts_dir / "data_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert "train.csv" in manifest
    assert manifest["train.csv"]["rows"] == 1
    assert manifest["train.csv"]["columns"] == [
        "id",
        "product_uid",
        "product_title",
        "search_term",
        "relevance",
    ]
    assert "sha256" in manifest["train.csv"]
    assert "size_bytes" in manifest["train.csv"]

def test_check_data_multiline_csv(tmp_path, capsys):
    # Setup mock config and valid files with a multiline field
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train_file = data_dir / "train.csv"
    train_file.write_text(
        "id,product_uid,product_title,search_term,relevance\n1,1001,A,B,3.0\n"
    )

    test_file = data_dir / "test.csv"
    test_file.write_text("id,product_uid,product_title,search_term\n2,1002,C,D\n")

    attr_file = data_dir / "attributes.csv"
    attr_file.write_text("product_uid,name,value\n1001,Brand,Home Depot\n")

    desc_file = data_dir / "product_descriptions.csv"
    desc_file.write_text(
        "product_uid,product_description\n1001,\"A great product\\nwith multiple\\nlines\"\n1002,Another product\n"
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
data:
  train_path: "{train_file}"
  test_path: "{test_file}"
  attributes_path: "{attr_file}"
  product_descriptions_path: "{desc_file}"
model:
  name: "test"
""")

    artifacts_dir = tmp_path / "artifacts"
    
    # We need to create a config object
    class ConfigData:
        def __init__(self):
            self.train_path = str(train_file)
            self.test_path = str(test_file)
            self.attributes_path = str(attr_file)
            self.product_descriptions_path = str(desc_file)
            
    class Config:
        def __init__(self):
            self.data = ConfigData()
            
    config = Config()

    check_data(config, artifacts_dir)

    # Verify manifest generated
    manifest_path = artifacts_dir / "data_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert "product_descriptions.csv" in manifest
    # even though there are 5 lines in the file (1 header + 3 for first desc + 1 for second desc), 
    # there are only 2 rows of data
    assert manifest["product_descriptions.csv"]["rows"] == 2
