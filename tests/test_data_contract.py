import os

import pandas as pd

from home_depot_search.data_contract import DATA_DIR, verify_data_contract


def test_verify_data_contract_success():
    # Since we unpacked the data, it should successfully pass without AssertionError
    verify_data_contract()


def test_train_data_schema():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="ISO-8859-1")
    expected_cols = {"id", "product_uid", "product_title", "search_term", "relevance"}
    assert expected_cols.issubset(set(train.columns))


def test_test_data_schema():
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="ISO-8859-1")
    expected_cols = {"id", "product_uid", "product_title", "search_term"}
    assert expected_cols.issubset(set(test.columns))


def test_attributes_schema():
    attributes = pd.read_csv(
        os.path.join(DATA_DIR, "attributes.csv"), encoding="ISO-8859-1"
    )
    expected_cols = {"product_uid", "name", "value"}
    assert expected_cols.issubset(set(attributes.columns))


def test_product_descriptions_schema():
    descriptions = pd.read_csv(
        os.path.join(DATA_DIR, "product_descriptions.csv"), encoding="ISO-8859-1"
    )
    expected_cols = {"product_uid", "product_description"}
    assert expected_cols.issubset(set(descriptions.columns))
