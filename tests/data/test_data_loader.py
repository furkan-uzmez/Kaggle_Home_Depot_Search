import pytest
import pandas as pd
from pathlib import Path
from home_depot_search.data.data_loader import (
    load_train_data,
    load_test_data,
    load_product_descriptions,
    load_attributes
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

