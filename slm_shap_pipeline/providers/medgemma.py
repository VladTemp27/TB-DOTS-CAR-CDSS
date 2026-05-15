from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.processor() == "arm"


class MedGemmaProvider:
    """Synchronous llama-cpp-python provider for the local MedGemma GGUF model."""

    name = "medgemma"

    def __init__(
        self,
        model_path: str | Path = "models/medgemma-1.5-4b-it-IQ4_XS.gguf",
        n_ctx: int = 2048,
        max_tokens: int = 1536,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
    ) -> None:
        self._model_path = Path(model_path)
        self._n_ctx = n_ctx
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._repeat_penalty = repeat_penalty
        self._llm = None  # lazy: loaded on first generate()

    def _load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python not installed — run: pip install llama-cpp-python"
            ) from exc

        if not self._model_path.exists():
            raise RuntimeError(f"MedGemma model not found: {self._model_path.resolve()}")

        apple_silicon = _is_apple_silicon()
        n_threads = max(4, (os.cpu_count() or 4) // 2)
        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=self._n_ctx,
            n_batch=512,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            n_gpu_layers=-1,
            flash_attn=not apple_silicon,  # Metal builds have issues with flash attention
            verbose=False,
        )

    def generate(self, prompt: str) -> str:
        if self._llm is None:
            self._load()

        output = self._llm(  # type: ignore[misc]
            prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            repeat_penalty=self._repeat_penalty,
        )
        text = output["choices"][0]["text"].strip()
        if not text:
            raise RuntimeError("MedGemma returned an empty response")
        return text
