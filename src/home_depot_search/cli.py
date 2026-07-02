import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from home_depot_search.config import load_config
from home_depot_search.data.validation import EXPECTED_FILES, TRAIN_FILE, TRAIN_SCHEMA

CHUNK_SIZE = 100_000


def get_file_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def check_data(config, artifacts_dir: Path = None) -> None:
    """Check if data files exist and generate a manifest."""
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"

    data_paths = {
        "train.csv": Path(config.data.train_path),
        "test.csv": Path(config.data.test_path),
        "attributes.csv": Path(config.data.attributes_path),
        "product_descriptions.csv": Path(config.data.product_descriptions_path),
    }

    missing_files = []
    for expected_file in EXPECTED_FILES:
        path = data_paths[expected_file]
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        print(
            f"Error: Missing expected data files: {', '.join(missing_files)}",
            file=sys.stderr,
        )
        print(
            "Please download the Kaggle dataset and place them "
            "in the correct directories.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate schema for train.csv
    train_path = data_paths[TRAIN_FILE]
    try:
        df_train_header = pd.read_csv(train_path, nrows=0, encoding='ISO-8859-1')
        train_columns = df_train_header.columns.tolist()
        missing_columns = [col for col in TRAIN_SCHEMA if col not in train_columns]
        if missing_columns:
            print(
                f"Error: Missing expected columns in {TRAIN_FILE}: "
                f"{', '.join(missing_columns)}",
                file=sys.stderr,
            )
            sys.exit(1)
    except Exception as e:
        print(f"Error reading {train_path} for schema validation: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate manifest
    manifest = {}
    for filename, path in data_paths.items():
        try:
            # Try parsing to get columns
            df = pd.read_csv(path, nrows=0, encoding='ISO-8859-1')  # Read only header first to get columns
            columns = df.columns.tolist()

            # Efficient count without loading entire file into memory if it's large,
            # considering quoted newlines
            rows = sum(
                len(chunk)
                for chunk in pd.read_csv(path, usecols=[0], chunksize=CHUNK_SIZE, encoding='ISO-8859-1')
            )

            manifest[filename] = {
                "size_bytes": path.stat().st_size,
                "sha256": get_file_sha256(path),
                "columns": columns,
                "rows": rows,
            }
        except Exception as e:
            print(f"Error reading {path}: {e}", file=sys.stderr)
            sys.exit(1)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts_dir / "data_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print(f"Data validation successful. Manifest written to {manifest_path}")


def main(args=None):
    parser = argparse.ArgumentParser(description="Home Depot Search Relevance")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check data files and generate manifest",
    )

    parsed_args = parser.parse_args(args)

    try:
        config = load_config(parsed_args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    if parsed_args.check_data:
        check_data(config)


if __name__ == "__main__":
    main()
