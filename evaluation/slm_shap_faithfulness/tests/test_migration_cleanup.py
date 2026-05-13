from pathlib import Path

_REPO_ROOT = Path(__file__).parents[4]


def test_legacy_benchmark_paths_removed():
    assert not (_REPO_ROOT / "non_temporal" / "faithfulness").exists()
    assert not (_REPO_ROOT / "non-temporal" / "tests").exists()
