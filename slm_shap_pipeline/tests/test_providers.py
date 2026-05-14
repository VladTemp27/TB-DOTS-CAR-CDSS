import os
import pytest
from unittest.mock import patch, MagicMock

from slm_shap_pipeline.config import PipelineConfig
from slm_shap_pipeline.providers.base import SLMProvider, make_provider
from slm_shap_pipeline.providers.cli import GeminiCLIProvider
from slm_shap_pipeline.providers.api import GoogleAPIProvider
from slm_shap_pipeline.providers.medgemma import MedGemmaProvider


def test_provider_protocol_requires_name_and_generate():
    class Stub:
        name = "stub"
        def generate(self, prompt: str) -> str: return "ok"
    assert isinstance(Stub(), SLMProvider)


def test_cli_provider_invokes_subprocess():
    provider = GeminiCLIProvider(model="gemini-test", timeout=60)
    mock_result = MagicMock(returncode=0, stdout="cli response", stderr="")
    with patch("slm_shap_pipeline.providers.cli.subprocess.run",
               return_value=mock_result) as mock_run:
        out = provider.generate("hello")
    assert out == "cli response"
    args, kwargs = mock_run.call_args
    assert args[0] == ["gemini", "--model", "gemini-test"]
    assert kwargs["input"] == "hello"
    assert kwargs["timeout"] == 60


def test_cli_provider_raises_on_failure():
    provider = GeminiCLIProvider(model="gemini-test")
    mock_result = MagicMock(returncode=1, stdout="", stderr="auth failed")
    with patch("slm_shap_pipeline.providers.cli.subprocess.run",
               return_value=mock_result):
        with pytest.raises(RuntimeError, match="Gemini CLI"):
            provider.generate("x")


def test_cli_provider_name():
    assert GeminiCLIProvider().name == "gemini-cli"


def test_api_provider_calls_genai_sdk():
    provider = GoogleAPIProvider(api_key="fake-key", model="gemini-test")
    fake_response = MagicMock(text="api response")
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    with patch.object(provider, "_client", fake_client):
        out = provider.generate("hello")
    assert out == "api response"
    fake_client.models.generate_content.assert_called_once()
    call_kwargs = fake_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-test"
    assert call_kwargs["contents"] == "hello"


def test_api_provider_name():
    p = GoogleAPIProvider(api_key="x", model="m")
    assert p.name == "google-api"


def test_api_provider_raises_on_empty_response():
    provider = GoogleAPIProvider(api_key="fake-key", model="gemini-test")
    fake_response = MagicMock(text="")
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    with patch.object(provider, "_client", fake_client):
        with pytest.raises(RuntimeError, match="empty"):
            provider.generate("hello")


def test_factory_returns_cli_provider_by_default():
    cfg = PipelineConfig()
    p = make_provider(cfg)
    assert isinstance(p, GeminiCLIProvider)
    assert p.name == "gemini-cli"


def test_factory_returns_api_provider_when_selected(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    cfg = PipelineConfig(provider="api")
    p = make_provider(cfg)
    assert isinstance(p, GoogleAPIProvider)
    assert p.name == "google-api"


def test_factory_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    cfg = PipelineConfig(provider="api")
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        make_provider(cfg)


def test_factory_rejects_unknown_provider():
    cfg = PipelineConfig(provider="bogus")
    with pytest.raises(ValueError, match="Unknown provider"):
        make_provider(cfg)


def test_medgemma_provider_name():
    p = MedGemmaProvider(model_path="models/medgemma-1.5-4b-it-IQ4_XS.gguf")
    assert p.name == "medgemma"


def test_medgemma_provider_raises_when_llama_cpp_missing(tmp_path):
    # Simulate llama_cpp not installed
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00")
    provider = MedGemmaProvider(model_path=gguf)
    with patch.dict("sys.modules", {"llama_cpp": None}):
        with pytest.raises(RuntimeError, match="llama-cpp-python"):
            provider.generate("hello")


def test_medgemma_provider_raises_when_model_missing(tmp_path):
    provider = MedGemmaProvider(model_path=tmp_path / "nonexistent.gguf")
    fake_llama_cpp = MagicMock()
    fake_llama_cpp.Llama = MagicMock()
    with patch.dict("sys.modules", {"llama_cpp": fake_llama_cpp}):
        with pytest.raises(RuntimeError, match="not found"):
            provider.generate("hello")


def test_medgemma_provider_generate_calls_llm(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00")
    provider = MedGemmaProvider(model_path=gguf)

    fake_llm = MagicMock(return_value={"choices": [{"text": " medgemma response "}]})
    fake_Llama = MagicMock(return_value=fake_llm)
    fake_llama_cpp = MagicMock()
    fake_llama_cpp.Llama = fake_Llama

    with patch.dict("sys.modules", {"llama_cpp": fake_llama_cpp}):
        result = provider.generate("test prompt")

    assert result == "medgemma response"
    fake_llm.assert_called_once_with(
        "test prompt",
        max_tokens=provider._max_tokens,
        temperature=provider._temperature,
        top_p=provider._top_p,
        repeat_penalty=provider._repeat_penalty,
    )


def test_medgemma_provider_raises_on_empty_response(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00")
    provider = MedGemmaProvider(model_path=gguf)

    fake_llm = MagicMock(return_value={"choices": [{"text": "   "}]})
    fake_Llama = MagicMock(return_value=fake_llm)
    fake_llama_cpp = MagicMock()
    fake_llama_cpp.Llama = fake_Llama

    with patch.dict("sys.modules", {"llama_cpp": fake_llama_cpp}):
        with pytest.raises(RuntimeError, match="empty"):
            provider.generate("hello")


def test_factory_returns_medgemma_provider(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00")
    cfg = PipelineConfig(provider="medgemma", medgemma_model_path=gguf)
    p = make_provider(cfg)
    assert isinstance(p, MedGemmaProvider)
    assert p.name == "medgemma"
