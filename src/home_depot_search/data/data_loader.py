import pandas as pd
from pathlib import Path
from typing import Union

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
