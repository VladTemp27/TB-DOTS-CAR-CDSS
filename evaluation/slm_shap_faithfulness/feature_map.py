import json
from pathlib import Path

_ALIAS_PATH = Path(__file__).parent / "feature_aliases.json"


def _load_aliases() -> dict[str, str]:
    try:
        with open(_ALIAS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"feature_aliases.json not found at {_ALIAS_PATH}. "
            "Ensure the file is present in the package directory."
        ) from None
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"feature_aliases.json at {_ALIAS_PATH} is not valid JSON: {exc}"
        ) from exc


_DEFAULT_ALIASES: dict[str, str] = _load_aliases()


def canonicalize_feature(raw_name: str, alias_map: dict[str, str] | None = None) -> str | None:
    """Return canonical feature name from alias_map, or None if raw_name is not recognized."""
    if alias_map is None:
        alias_map = _DEFAULT_ALIASES
    return alias_map.get(raw_name)
