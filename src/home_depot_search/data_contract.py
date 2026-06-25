import os

import pandas as pd

DATA_DIR = "data/home-depot-product-search-relevance"


def verify_data_contract():
    required_files = [
        "train.csv",
        "test.csv",
        "attributes.csv",
        "product_descriptions.csv",
    ]
    for file in required_files:
        path = os.path.join(DATA_DIR, file)
        assert os.path.exists(path), f"Missing data file: {path}"

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="ISO-8859-1")
    assert all(
        col in train.columns
        for col in ["id", "product_uid", "product_title", "search_term", "relevance"]
    ), "Train data schema mismatch"

    print("Data contract verified successfully.")


if __name__ == "__main__":
    verify_data_contract()
