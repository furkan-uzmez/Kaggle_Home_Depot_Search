import pandas as pd
from pathlib import Path
from typing import Union

from home_depot_search.data.text_cleaning import clean_text

def load_train_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load train.csv enforcing data types.
    
    Args:
        filepath: Path to train.csv
        
    Returns:
        DataFrame with columns: id, product_uid, product_title, search_term, relevance
    """
    dtypes = {
        'id': 'Int32',  # Using nullable Int32 just in case
        'product_uid': 'Int32',
        'product_title': 'string',
        'search_term': 'string',
        'relevance': 'float32'
    }
    return pd.read_csv(filepath, dtype=dtypes, encoding='ISO-8859-1')

def load_test_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load test.csv enforcing data types.
    
    Args:
        filepath: Path to test.csv
        
    Returns:
        DataFrame with columns: id, product_uid, product_title, search_term
    """
    dtypes = {
        'id': 'Int32',
        'product_uid': 'Int32',
        'product_title': 'string',
        'search_term': 'string'
    }
    return pd.read_csv(filepath, dtype=dtypes, encoding='ISO-8859-1')

def load_product_descriptions(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load product_descriptions.csv enforcing data types.
    
    Args:
        filepath: Path to product_descriptions.csv
        
    Returns:
        DataFrame with columns: product_uid, product_description
    """
    dtypes = {
        'product_uid': 'Int32',
        'product_description': 'string'
    }
    return pd.read_csv(filepath, dtype=dtypes, encoding='ISO-8859-1')

def merge_product_descriptions(
    df: pd.DataFrame,
    descriptions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge product descriptions into a DataFrame on product_uid.

    Performs a left join so that row count and order of the input DataFrame
    are preserved. Products without a matching description receive an empty
    string.

    Args:
        df: DataFrame with a 'product_uid' column (train or test).
        descriptions: DataFrame with 'product_uid' and 'product_description'.

    Returns:
        DataFrame with an added 'product_description' column. Row count
        and index of the input are unchanged.
    """
    if descriptions["product_uid"].duplicated().any():
        raise ValueError(
            "product_description product_uid must be unique before merging."
        )

    merged = pd.merge(
        df,
        descriptions,
        on="product_uid",
        how="left",
        validate="m:1",
    )
    merged["product_description"] = merged["product_description"].fillna("")
    return merged


def aggregate_attributes(attributes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate attribute name-value pairs into a single text field per product_uid.

    Rows with a null product_uid are dropped because they cannot be matched
    to any product. The remaining rows are grouped by product_uid and each
    attribute is formatted as ``name: value``, joined by `` | ``.

    Args:
        attributes: DataFrame with columns product_uid, name, value.

    Returns:
        DataFrame with columns product_uid and product_attributes (string).
        One row per product_uid that had at least one non-null attribute row.
    """
    valid = attributes.dropna(subset=["product_uid"]).copy()
    valid["name"] = valid["name"].fillna("")
    valid["value"] = valid["value"].fillna("")
    valid["product_attributes"] = valid["name"] + ": " + valid["value"]
    aggregated = (
        valid.groupby("product_uid", observed=True)["product_attributes"]
        .agg(" | ".join)
        .reset_index()
    )
    aggregated["product_attributes"] = aggregated["product_attributes"].astype("string")
    return aggregated


def build_merged_dataset(
    train_file: Union[str, Path],
    test_file: Union[str, Path],
    descriptions_file: Union[str, Path],
    attributes_file: Union[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files and build the merged train/test datasets.

    This function orchestrates the full Phase 2 merge pipeline:

    1. Load raw CSV files with dtype enforcement.
    2. Apply deterministic text cleanup (Unicode/whitespace normalisation).
    3. Merge product descriptions on ``product_uid``.
    4. Aggregate attribute name-value pairs and merge on ``product_uid``.
    5. Produce ``product_text_raw`` = title + description + attributes.

    Args:
        train_file: Path to train.csv.
        test_file: Path to test.csv.
        descriptions_file: Path to product_descriptions.csv.
        attributes_file: Path to attributes.csv.

    Returns:
        Tuple of (train_df, test_df), each with columns:
            search_term_raw, product_title_raw,
            product_description_raw, attribute_text_raw,
            product_text_raw.
        The train_df additionally contains the ``relevance`` column.
        Row counts of the original CSV files are preserved.
    """
    # 1. Load raw data
    train = load_train_data(train_file)
    test = load_test_data(test_file)
    descriptions = load_product_descriptions(descriptions_file)
    attributes = load_attributes(attributes_file)

    # 2. Deterministic text cleanup (pre-split safe)
    for frame in (train, test):
        frame["search_term_raw"] = frame["search_term"].apply(clean_text)
        frame["product_title_raw"] = frame["product_title"].apply(clean_text)

    # 3. Merge product descriptions
    train = merge_product_descriptions(train, descriptions)
    test = merge_product_descriptions(test, descriptions)

    # 4. Aggregate attributes and merge
    attribute_text = aggregate_attributes(attributes)
    train = pd.merge(
        train, attribute_text, on="product_uid", how="left", validate="m:1"
    )
    test = pd.merge(
        test, attribute_text, on="product_uid", how="left", validate="m:1"
    )
    train["attribute_text_raw"] = train["product_attributes"].fillna("")
    test["attribute_text_raw"] = test["product_attributes"].fillna("")

    # 5. Build combined product text
    train["product_text_raw"] = (
        train["product_title_raw"]
        + " "
        + train["product_description"]
        + " "
        + train["attribute_text_raw"]
    ).str.strip()
    test["product_text_raw"] = (
        test["product_title_raw"]
        + " "
        + test["product_description"]
        + " "
        + test["attribute_text_raw"]
    ).str.strip()

    return train, test


def load_attributes(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load attributes.csv enforcing data types.
    
    Note: attributes.csv contains null values for product_uid in some rows.
    These are loaded as pd.NA using the nullable 'Int32' type.
    
    Args:
        filepath: Path to attributes.csv
        
    Returns:
        DataFrame with columns: product_uid, name, value
    """
    dtypes = {
        'product_uid': 'Int32',
        'name': 'string',
        'value': 'string'
    }
    return pd.read_csv(filepath, dtype=dtypes, encoding='ISO-8859-1')
