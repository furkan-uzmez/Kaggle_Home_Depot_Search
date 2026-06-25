from typing import Final

# Expected Kaggle data files
TRAIN_FILE: Final[str] = "train.csv"
TEST_FILE: Final[str] = "test.csv"
ATTRIBUTES_FILE: Final[str] = "attributes.csv"
PRODUCT_DESCRIPTIONS_FILE: Final[str] = "product_descriptions.csv"

EXPECTED_FILES: Final[set[str]] = {
    TRAIN_FILE,
    TEST_FILE,
    ATTRIBUTES_FILE,
    PRODUCT_DESCRIPTIONS_FILE,
}

# Expected schema for train.csv
TRAIN_SCHEMA: Final[list[str]] = [
    "id",
    "product_uid",
    "product_title",
    "search_term",
    "relevance",
]
