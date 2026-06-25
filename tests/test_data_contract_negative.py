import pytest


def test_data_contract_missing_file(tmp_path, monkeypatch):
    from home_depot_search.data_contract import verify_data_contract

    monkeypatch.setattr("home_depot_search.data_contract.DATA_DIR", str(tmp_path))

    with pytest.raises(AssertionError, match="Missing data file:"):
        verify_data_contract()
