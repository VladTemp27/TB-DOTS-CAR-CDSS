"""I/O utilities for writing benchmark output artifacts."""
import json
from pathlib import Path


def write_results(output_dir: str | Path, results: dict) -> None:
    """Write benchmark results dict as JSON to output_dir/results.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))


def read_results(results_path: str | Path) -> dict:
    """Read a previously written results JSON file."""
    return json.loads(Path(results_path).read_text())
