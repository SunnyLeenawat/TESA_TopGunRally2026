"""OpenAI-compatible LLM client that never crashes the agent.

The organizer gateway speaks the OpenAI chat-completions API.  This module uses
``httpx`` directly (already a dependency of the practice kit) instead of adding
the ``openai`` package.  The behaviour is the same as the hint notebook's
``ask_json``:

* request a JSON answer with ``response_format`` / ``json_schema``;
* return ``None`` on timeout, 429, bad key, connection failure or broken JSON;
* remember the reason in ``last_error`` for the decision log;
* never throw from ``ask_json``.

The gateway limits are 2 concurrent requests and 20 requests/minute per team.
The starter asks at most once per round, so no retry loop is used here.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from starter.config import Settings

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)


def parse_json(text: Any) -> Any | None:
    """Parse JSON even when the model wraps it in prose, fences or think tags."""
    if not isinstance(text, str):
        return None
    cleaned = _THINK_BLOCK.sub("", text).strip()
    cleaned = _FENCE.sub("", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Last resort: take the outermost {...} or [...] block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


class LLMClient:
    """Small, non-throwing client for the organizer's ``event-llm`` gateway."""

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model
        self.api_key = settings.llm_api_key or settings.team_key
        self.timeout = settings.llm_timeout
        self.thinking = settings.llm_thinking
        self.max_tokens = settings.llm_max_tokens
        self.enabled = bool(settings.use_llm and self.base_url and self.model)
        self.last_error: str | None = None

    def ask_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> Any | None:
        """Ask for a JSON answer.  Returns a parsed object or ``None``."""
        self.last_error = None
        if not self.enabled:
            self.last_error = "LLM disabled (USE_LLM=0 or missing URL/model)"
            return None

        effective_timeout = min(
            self.timeout, float(timeout) if timeout is not None else self.timeout
        )
        effective_timeout = max(1.0, effective_timeout)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
            "max_tokens": int(max_tokens or self.max_tokens),
            "reasoning_effort": "medium" if self.thinking else "none",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            with httpx.Client(timeout=effective_timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
        except httpx.TimeoutException:
            self.last_error = f"timeout after {effective_timeout:.1f}s"
            return None
        except httpx.HTTPError as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

        if response.status_code >= 400:
            self.last_error = self._describe_http_error(response)
            return None

        try:
            body = response.json()
            choice = body["choices"][0]
            text = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self.last_error = f"malformed gateway response: {type(exc).__name__}"
            return None

        if choice.get("finish_reason") == "length":
            # The JSON is probably cut off.  Try to parse anyway; if it fails,
            # report the real reason instead of a generic decode error.
            parsed = parse_json(text)
            if parsed is None:
                self.last_error = "finish_reason=length (max_tokens too small)"
            return parsed

        parsed = parse_json(text)
        if parsed is None:
            preview = str(text)[:160].replace("\n", " ")
            self.last_error = f"invalid JSON: {preview}"
        return parsed

    @staticmethod
    def _describe_http_error(response: httpx.Response) -> str:
        code = f"HTTP {response.status_code}"
        try:
            body = response.json()
            error = body.get("error", body) if isinstance(body, dict) else {}
            if isinstance(error, dict):
                name = error.get("code") or error.get("type") or ""
                message = error.get("message") or ""
                if name or message:
                    return f"{code} {name}: {message}".strip()
        except ValueError:
            pass
        retry_after = response.headers.get("Retry-After")
        suffix = f" retry_after={retry_after}s" if retry_after else ""
        return f"{code}{suffix}: {response.text[:160]}"
