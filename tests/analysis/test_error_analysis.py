import numpy as np
import pandas as pd

from home_depot_search.analysis.error_analysis import run_error_analysis


def test_run_error_analysis(tmp_path):
    y_true = np.array([1.0, 2.0, 2.5, 3.0, 1.5, 2.0])
    y_pred = np.array([1.1, 2.1, 2.4, 2.8, 1.6, 2.2])
    df = pd.DataFrame({
        "search_term_raw": ["a b", "c d e", "f", "g h", "i j k", "l"],
        "product_title_raw": ["title1", "title2", "title3", "title4", "title5", "title6"],
        "product_description": ["desc1", "desc2", "desc3", "desc4", "desc5", "desc6"],
    })

    report_path = run_error_analysis(y_true, y_pred, df, output_dir=str(tmp_path))
    path = tmp_path / "error_analysis_report.md"
    assert path.exists()
    text = path.read_text()
    assert "Error Analysis" in text or "Error Analysis Report" in text
    assert "RMSE" in text
