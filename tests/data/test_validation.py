from home_depot_search.data.validation import (
    ATTRIBUTES_FILE,
    EXPECTED_FILES,
    PRODUCT_DESCRIPTIONS_FILE,
    TEST_FILE,
    TRAIN_FILE,
    TRAIN_SCHEMA,
)


def test_expected_files_contain_all_constants():
    """Test that the EXPECTED_FILES set contains all the individual file constants."""
    assert TRAIN_FILE in EXPECTED_FILES
    assert TEST_FILE in EXPECTED_FILES
    assert ATTRIBUTES_FILE in EXPECTED_FILES
    assert PRODUCT_DESCRIPTIONS_FILE in EXPECTED_FILES
    assert len(EXPECTED_FILES) == 4


def test_train_schema_contains_required_columns():
    """Test that TRAIN_SCHEMA contains the exactly required columns in order."""
    expected_columns = [
        "id",
        "product_uid",
        "product_title",
        "search_term",
        "relevance",
    ]
    assert TRAIN_SCHEMA == expected_columns
