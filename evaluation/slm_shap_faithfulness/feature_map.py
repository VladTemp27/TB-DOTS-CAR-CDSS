import json
from pathlib import Path

_ALIAS_PATH = Path(__file__).parent / "feature_aliases.json"


def _load_aliases() -> dict[str, str]:
    with open(_ALIAS_PATH) as f:
        return json.load(f)


_DEFAULT_ALIASES: dict[str, str] = _load_aliases()


def canonicalize_feature(raw_name: str, alias_map: dict[str, str] | None = None) -> str | None:
    if alias_map is None:
        alias_map = _DEFAULT_ALIASES
    return alias_map.get(raw_name)
