import json

from home_depot_search.utils.logging import (
    RunLogger,
    get_timestamp,
    init_experiment_csv,
)


def test_get_timestamp_returns_string():
    ts = get_timestamp()
    assert isinstance(ts, str)


def test_init_experiment_csv_creates_file(tmp_path):
    csv_path = init_experiment_csv(str(tmp_path))
    assert csv_path.exists()


def test_init_experiment_csv_appends_header(tmp_path):
    path1 = init_experiment_csv(str(tmp_path))
    path2 = init_experiment_csv(str(tmp_path))
    assert path1 == path2
    lines = path1.read_text().strip().split("\n")
    assert len(lines) == 1


def test_run_logger_log_metadata(tmp_path):
    logger = RunLogger(run_id="test_001", log_dir=str(tmp_path))
    logger.log_metadata("model_name", "xgboost")
    assert logger.metadata["model_name"] == "xgboost"


def test_run_logger_log_fold_metrics(tmp_path):
    logger = RunLogger(run_id="test_001", log_dir=str(tmp_path))
    logger.log_fold_metrics(fold=0, rmse=0.5, clipped_rmse=0.4)
    assert len(logger.fold_metrics) == 1
    assert logger.fold_metrics[0] == {"fold": 0, "rmse": 0.5, "clipped_rmse": 0.4}


def test_run_logger_save_run_json(tmp_path):
    logger = RunLogger(run_id="test_001", log_dir=str(tmp_path))
    logger.log_metadata("model_name", "xgboost")
    logger.log_fold_metrics(fold=0, rmse=0.5, clipped_rmse=0.4)
    logger.save_run_json()
    json_path = tmp_path / "runs" / "test_001.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["model_name"] == "xgboost"
    assert len(data["fold_metrics"]) == 1
