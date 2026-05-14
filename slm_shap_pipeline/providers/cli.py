from __future__ import annotations

import subprocess


class GeminiCLIProvider:
    name = "gemini-cli"

    def __init__(self, model: str = "gemini-2.5-pro-preview-05-06", timeout: int = 120) -> None:
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        result = subprocess.run(
            ["gemini", "--model", self.model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Gemini CLI failed (exit {result.returncode}): {result.stderr}")
        return result.stdout.strip()
