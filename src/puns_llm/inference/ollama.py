import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaClient:
    def __init__(self, host: str, timeout_seconds: int):
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request(self, endpoint: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.host}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Ollama HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f"Cannot reach Ollama at {self.host}: {error}") from error

    def model_digest(self, name: str) -> str:
        models = self.request("/api/tags").get("models", [])

        for model in models:
            if model.get("name") == name:
                return model["digest"]

        raise ValueError(f"Model not available in Ollama: {name}")

    def chat(
        self,
        model: str,
        messages: list[dict],
        schema: dict,
        options: dict,
        think: bool | None,
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "options": options,
        }

        if think is not None:
            payload["think"] = think

        return self.request("/api/chat", payload)