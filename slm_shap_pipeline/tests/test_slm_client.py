import json
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from slm_shap_pipeline.slm_client import call_slm, make_cache_key


class _StubProvider:
    name = "stub"
    def __init__(self, response: str = "stub response"):
        self.response = response
        self.calls = 0
    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.response


def _key():
    return make_cache_key("P001", "sighted", 12, "prompt", "mhash", "shash", "thash", "fhash")


def test_make_cache_key_is_deterministic():
    k1 = make_cache_key("P001", "sighted", 12, "prompt", "mhash", "shash", "thash", "fhash")
    k2 = make_cache_key("P001", "sighted", 12, "prompt", "mhash", "shash", "thash", "fhash")
    assert k1 == k2


def test_make_cache_key_differs_on_condition():
    k1 = make_cache_key("P001", "sighted", 12, "prompt", "mhash", "shash", "thash", "fhash")
    k2 = make_cache_key("P001", "blind", 12, "prompt", "mhash", "shash", "thash", "fhash")
    assert k1 != k2


def test_make_cache_key_differs_on_prompt():
    k1 = make_cache_key("P001", "sighted", 12, "prompt A", "mhash", "shash", "thash", "fhash")
    k2 = make_cache_key("P001", "sighted", 12, "prompt B", "mhash", "shash", "thash", "fhash")
    assert k1 != k2


def test_make_cache_key_ignores_provider_name():
    """Cache stays provider-agnostic: a CLI-warmed cache is reusable under API."""
    import inspect
    sig = inspect.signature(make_cache_key)
    assert "provider" not in sig.parameters


def test_cache_hit_short_circuits_provider(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = _key()
    (cache_dir / f"{key}.json").write_text(
        json.dumps({"response": "cached", "provider": "gemini-cli"})
    )
    provider = _StubProvider(response="should not be called")
    result = call_slm("prompt", key, provider, cache_dir=cache_dir)
    assert result == "cached"
    assert provider.calls == 0


def test_cache_miss_calls_provider_and_saves(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = _key()
    provider = _StubProvider(response="fresh response")
    result = call_slm("prompt", key, provider, cache_dir=cache_dir)
    assert result == "fresh response"
    assert provider.calls == 1
    cache_file = cache_dir / f"{key}.json"
    cached = json.loads(cache_file.read_text())
    assert cached["response"] == "fresh response"
    assert cached["provider"] == "stub"


def test_dry_run_skips_provider(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = _key()
    provider = _StubProvider()
    result = call_slm("prompt", key, provider, cache_dir=cache_dir, dry_run=True)
    assert provider.calls == 0
    assert len(result) > 0


def test_provider_error_propagates(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = _key()
    bad = MagicMock()
    bad.name = "broken"
    bad.generate.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        call_slm("prompt", key, bad, cache_dir=cache_dir)
    assert not (cache_dir / f"{key}.json").exists()  # don't poison cache on failure
