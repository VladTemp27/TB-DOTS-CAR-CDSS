"""I/O utilities for writing benchmark output artifacts."""
import json
from pathlib import Path


def write_results(output_dir: str | Path, results: dict) -> None:
    """Write benchmark results dict as JSON to output_dir/results.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def read_results(results_path: str | Path) -> dict:
    """Read a previously written results JSON file."""
    results_path = Path(results_path)
    try:
        return json.loads(results_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError(f"Results file not found: {results_path}") from None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {results_path}: {exc}") from exc
