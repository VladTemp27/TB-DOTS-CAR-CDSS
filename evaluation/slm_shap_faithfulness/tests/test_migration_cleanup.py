from pathlib import Path


def test_legacy_benchmark_paths_removed():
    assert not Path("non_temporal/faithfulness").exists()
    assert not Path("non-temporal/tests").exists()
