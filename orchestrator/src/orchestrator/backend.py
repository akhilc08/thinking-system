"""ModelBackend protocol and the OpenRouter implementation (spec §5.1).
All orchestrator pipeline stages are written against ModelBackend, never
against a specific provider — see also claude_code_backend.py for the
second implementation.
"""
import time
from typing import Protocol

import requests

from config import OPENROUTER_BASE_URL


class ModelBackend(Protocol):
    def call(self, role: str, system_prompt: str, user_prompt: str) -> str: ...


class BackendError(Exception):
    pass


def retry_with_backoff(fn, max_attempts: int = 3, base_delay: float = 1.0):
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except BackendError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc


class OpenRouterBackend:
    def __init__(self, model: str, api_key: str, base_url: str = OPENROUTER_BASE_URL, timeout: float = 60.0):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def call(self, role: str, system_prompt: str, user_prompt: str) -> str:
        def _do_call() -> str:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise BackendError(f"OpenRouter call failed for role={role}: {response.status_code} {response.text}")
            data = response.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                raise BackendError(f"Unexpected OpenRouter response shape: {data}") from exc

        return retry_with_backoff(_do_call)
