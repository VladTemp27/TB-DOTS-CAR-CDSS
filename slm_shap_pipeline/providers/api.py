from __future__ import annotations


class GoogleAPIProvider:
    name = "google-api"

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro") -> None:
        self.model = model
        self._api_key = api_key
        self._client = None   # created lazily on first generate() call

    def generate(self, prompt: str) -> str:
        if self._client is None:
            from google import genai  # type: ignore
            self._client = genai.Client(api_key=self._api_key)
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        text = getattr(response, "text", "") or ""
        if not text.strip():
            raise RuntimeError("GoogleAPIProvider received empty response")
        return text.strip()
