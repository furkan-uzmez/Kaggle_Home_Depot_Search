import pandas as pd

from home_depot_search.models.deberta_submission import build_pair_text


def test_build_pair_text_matches_transformer_training_contract():
    test = pd.DataFrame(
        {
            "search_term": ["12\" bracket"],
            "product_title": ["A 12\" BRACKET"],
            "product_uid": [1],
        }
    )
    descriptions = pd.DataFrame(
        {
            "product_uid": [1],
            "product_description": ["Fits  12\" spaces"],
        }
    )
    attributes = pd.DataFrame(
        {
            "product_uid": [1],
            "name": ["Color"],
            "value": ["Black"],
        }
    )

    assert build_pair_text(test, descriptions, attributes).tolist() == [
        "12 in bracket [SEP] a 12 in bracket fits 12 in spaces color: black"
    ]
