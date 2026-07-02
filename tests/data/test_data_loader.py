import pytest
import pandas as pd
from pathlib import Path
from home_depot_search.data.data_loader import (
    load_train_data,
    load_test_data,
    load_product_descriptions,
    load_attributes,
    merge_product_descriptions,
    aggregate_attributes,
    build_merged_dataset,
)

@pytest.fixture
def sample_train_csv(tmp_path):
    filepath = tmp_path / "train.csv"
    data = (
        "id,product_uid,product_title,search_term,relevance\n"
        "2,100001,Simpson Strong-Tie 12-Gauge Angle,angle bracket,3.0\n"
        "3,100001,Simpson Strong-Tie 12-Gauge Angle,l bracket,2.5\n"
    )
    filepath.write_text(data, encoding='ISO-8859-1')
    return filepath

@pytest.fixture
def sample_test_csv(tmp_path):
    filepath = tmp_path / "test.csv"
    data = (
        "id,product_uid,product_title,search_term\n"
        "1,100001,Simpson Strong-Tie 12-Gauge Angle,angle bracket\n"
        "4,100001,Simpson Strong-Tie 12-Gauge Angle,metal l brackets\n"
    )
    filepath.write_text(data, encoding='ISO-8859-1')
    return filepath

@pytest.fixture
def sample_descriptions_csv(tmp_path):
    filepath = tmp_path / "product_descriptions.csv"
    data = (
        "product_uid,product_description\n"
        "100001,Not only do angles make joints stronger... \n"
        "100002,BEHR Premium Textured DECKOVER is an innovative... \n"
    )
    filepath.write_text(data, encoding='ISO-8859-1')
    return filepath

@pytest.fixture
def sample_attributes_csv(tmp_path):
    filepath = tmp_path / "attributes.csv"
    data = (
        "product_uid,name,value\n"
        "100001,Bullet01,Versatile connector for various 90 connections and home repair projects.\n"
        ",,\n" # testing missing values
        "100001,Bullet02,Stronger than angled nailing or screw fastening alone.\n"
    )
    filepath.write_text(data, encoding='ISO-8859-1')
    return filepath

def test_load_train_data(sample_train_csv):
    df = load_train_data(sample_train_csv)
    assert len(df) == 2
    assert list(df.columns) == ['id', 'product_uid', 'product_title', 'search_term', 'relevance']
    assert df['id'].dtype == 'Int32'
    assert df['product_uid'].dtype == 'Int32'
    assert df['product_title'].dtype == 'string'
    assert df['search_term'].dtype == 'string'
    assert df['relevance'].dtype == 'float32'
    assert df.loc[0, 'relevance'] == 3.0

def test_load_test_data(sample_test_csv):
    df = load_test_data(sample_test_csv)
    assert len(df) == 2
    assert list(df.columns) == ['id', 'product_uid', 'product_title', 'search_term']
    assert df['id'].dtype == 'Int32'
    assert df['product_uid'].dtype == 'Int32'
    assert df['product_title'].dtype == 'string'
    assert df['search_term'].dtype == 'string'

def test_load_product_descriptions(sample_descriptions_csv):
    df = load_product_descriptions(sample_descriptions_csv)
    assert len(df) == 2
    assert list(df.columns) == ['product_uid', 'product_description']
    assert df['product_uid'].dtype == 'Int32'
    assert df['product_description'].dtype == 'string'

def test_load_attributes(sample_attributes_csv):
    df = load_attributes(sample_attributes_csv)
    assert len(df) == 3
    assert list(df.columns) == ['product_uid', 'name', 'value']
    assert df['product_uid'].dtype == 'Int32'
    assert df['name'].dtype == 'string'
    assert df['value'].dtype == 'string'
    
    # check missing value handling
    assert pd.isna(df.loc[1, 'product_uid'])
    assert pd.isna(df.loc[1, 'name'])
    assert pd.isna(df.loc[1, 'value'])

def test_merge_product_descriptions_preserves_row_count(sample_train_csv, sample_descriptions_csv):
    train = load_train_data(sample_train_csv)
    desc = load_product_descriptions(sample_descriptions_csv)
    merged = merge_product_descriptions(train, desc)
    assert len(merged) == len(train)
    assert list(merged.columns) == list(train.columns) + ['product_description']

def test_merge_product_descriptions_correct_description(sample_train_csv, sample_descriptions_csv):
    train = load_train_data(sample_train_csv)
    desc = load_product_descriptions(sample_descriptions_csv)
    merged = merge_product_descriptions(train, desc)
    assert merged.loc[0, 'product_description'] == (
        "Not only do angles make joints stronger... "
    )

def test_merge_product_descriptions_empty_for_missing(sample_test_csv, sample_descriptions_csv):
    test = load_test_data(sample_test_csv)
    desc = load_product_descriptions(sample_descriptions_csv)
    merged = merge_product_descriptions(test, desc)
    assert merged['product_description'].dtype == 'string'
    # product_uid=100001 has a description; all matched rows should be non-empty
    assert len(merged[merged['product_description'] == ""]) == 0
    # product_uid=100002 is in descriptions but not in test; should be empty string still

def test_merge_product_descriptions_test_data(sample_test_csv, sample_descriptions_csv):
    test = load_test_data(sample_test_csv)
    desc = load_product_descriptions(sample_descriptions_csv)
    merged = merge_product_descriptions(test, desc)
    assert len(merged) == len(test)
    assert 'product_description' in merged.columns
    assert merged['product_description'].dtype == 'string'
    # verify the existing row gets the correct description
    assert "angles" in merged.loc[0, 'product_description']

def test_aggregate_attributes_basic(sample_attributes_csv):
    attributes = load_attributes(sample_attributes_csv)
    aggregated = aggregate_attributes(attributes)
    assert list(aggregated.columns) == ['product_uid', 'product_attributes']
    assert aggregated['product_attributes'].dtype == 'string'
    # Only product_uid=100001 has non-null rows
    assert len(aggregated) == 1
    assert aggregated.loc[0, 'product_uid'] == 100001

def test_aggregate_attributes_text_format(sample_attributes_csv):
    attributes = load_attributes(sample_attributes_csv)
    aggregated = aggregate_attributes(attributes)
    text = aggregated.loc[0, 'product_attributes']
    assert "Bullet01: Versatile" in text
    assert "Bullet02: Stronger" in text
    assert " | " in text

def test_aggregate_attributes_drops_null_product_uid(sample_attributes_csv):
    attributes = load_attributes(sample_attributes_csv)
    aggregated = aggregate_attributes(attributes)
    # The second row of the fixture has null product_uid (",,\n") and must be dropped
    assert len(aggregated) == 1

@pytest.fixture
def all_csv_files(tmp_path):
    """Create all four raw CSV files in tmp_path."""
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "id,product_uid,product_title,search_term,relevance\n"
        "1,100001,Simpson Strong-Tie 12-Gauge Angle,angle bracket,3.0\n"
        "2,100002,BEHR DECKOVER,deck paint,2.0\n",
        encoding="ISO-8859-1",
    )
    test_csv = tmp_path / "test.csv"
    test_csv.write_text(
        "id,product_uid,product_title,search_term\n"
        "3,100001,Simpson Strong-Tie 12-Gauge Angle,metal bracket\n",
        encoding="ISO-8859-1",
    )
    desc_csv = tmp_path / "product_descriptions.csv"
    desc_csv.write_text(
        "product_uid,product_description\n"
        "100001,Not only do angles make joints stronger\n"
        "100002,BEHR Premium Textured DECKOVER deck coating\n",
        encoding="ISO-8859-1",
    )
    attr_csv = tmp_path / "attributes.csv"
    attr_csv.write_text(
        "product_uid,name,value\n"
        "100001,Color,Silver\n"
        "100001,Size,12-Gauge\n"
        "100002,Color,Red\n"
        ",,\n",
        encoding="ISO-8859-1",
    )
    return {
        "train": train_csv,
        "test": test_csv,
        "desc": desc_csv,
        "attr": attr_csv,
    }

def test_build_merged_dataset_row_count(all_csv_files):
    train, test = build_merged_dataset(
        all_csv_files["train"],
        all_csv_files["test"],
        all_csv_files["desc"],
        all_csv_files["attr"],
    )
    assert len(train) == 2
    assert len(test) == 1

def test_build_merged_dataset_columns(all_csv_files):
    train, test = build_merged_dataset(
        all_csv_files["train"],
        all_csv_files["test"],
        all_csv_files["desc"],
        all_csv_files["attr"],
    )
    expected_raw = [
        "search_term_raw",
        "product_title_raw",
        "product_description",
        "attribute_text_raw",
        "product_text_raw",
    ]
    for col in expected_raw:
        assert col in train.columns, f"Missing {col} in train"
        assert col in test.columns, f"Missing {col} in test"
    assert "relevance" in train.columns
    assert "relevance" not in test.columns

def test_build_merged_dataset_product_text_raw(all_csv_files):
    train, test = build_merged_dataset(
        all_csv_files["train"],
        all_csv_files["test"],
        all_csv_files["desc"],
        all_csv_files["attr"],
    )
    # product_text_raw = title + " " + description + " " + attributes
    text = train.loc[0, "product_text_raw"]
    assert "Simpson Strong-Tie 12-Gauge Angle" in text  # title
    assert "angles make joints stronger" in text  # description
    assert "Color: Silver" in text  # attribute
    assert "Size: 12-Gauge" in text  # attribute
    assert len(text) > 0

def test_build_merged_dataset_attribute_text(all_csv_files):
    """Verify attribute text format and null row filtering."""
    train, test = build_merged_dataset(
        all_csv_files["train"],
        all_csv_files["test"],
        all_csv_files["desc"],
        all_csv_files["attr"],
    )
    # product_uid=100002 has Color: Red attribute
    row = train.loc[train["product_uid"] == 100002].iloc[0]
    assert "Color: Red" in row["attribute_text_raw"]
    # product_text_raw should include title + description + attributes
    assert "BEHR DECKOVER" in row["product_text_raw"]
    assert "BEHR Premium Textured DECKOVER" in row["product_text_raw"]
    assert "Color: Red" in row["product_text_raw"]

