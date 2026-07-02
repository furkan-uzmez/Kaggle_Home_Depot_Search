import subprocess
import sys


def test_argparse_list_features():
    result = subprocess.run(
        [sys.executable, "run_experiments.py", "--list-features"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Available features:" in result.stdout
    assert "baseline-mean" in result.stdout


def test_argparse_list_models():
    result = subprocess.run(
        [sys.executable, "run_experiments.py", "--list-models"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Available models:" in result.stdout
    assert "baseline-mean" in result.stdout


def test_argparse_defaults():
    result = subprocess.run(
        [sys.executable, "run_experiments.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "--features" in result.stdout
    assert "--models" in result.stdout
    assert "--n-folds" in result.stdout
    assert "--seed" in result.stdout
    assert "--list-features" in result.stdout
    assert "--list-models" in result.stdout
