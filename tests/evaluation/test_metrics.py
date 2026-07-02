import numpy as np
import pytest

from home_depot_search.evaluation.metrics import clipped_rmse, rmse


def test_rmse_perfect():
    result = rmse([1, 2, 3], [1, 2, 3])
    assert result == pytest.approx(0.0)


def test_rmse_simple():
    result = rmse([1, 2, 3], [2, 3, 4])
    assert result == pytest.approx(1.0)


def test_clipped_rmse_clips_high_values():
    y_true = np.array([3.0, 3.0])
    y_pred = np.array([4.0, 5.0])
    result = clipped_rmse(y_true, y_pred, lower=1.0, upper=3.0)
    expected = rmse(np.array([3.0, 3.0]), np.array([3.0, 3.0]))
    assert result == pytest.approx(expected)


def test_clipped_rmse_clips_low_values():
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([0.0, -1.0])
    result = clipped_rmse(y_true, y_pred, lower=1.0, upper=3.0)
    expected = rmse(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert result == pytest.approx(expected)


def test_clipped_rmse_equivalent_within_bounds():
    y_true = [2.0, 2.5, 1.5]
    y_pred = [2.0, 2.0, 1.0]
    assert clipped_rmse(y_true, y_pred) == pytest.approx(rmse(y_true, y_pred))


def test_rmse_list_input():
    result = rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result == pytest.approx(0.0)
