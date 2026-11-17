"""Thin client for the organizer market API (v1).

Every endpoint, error code and payload follows ``hint/คู่มือ-API.md``.  The
client is intentionally small: it authenticates, throttles requests below the
5 requests/second limit, parses the common error envelope and raises typed
exceptions so the main loop can decide whether to retry, wait or skip a round.
"""

from __future__ import annotations

import time
from typing import Any, Self

import httpx


class MarketError(RuntimeError):
    """An API error with the organizer's ``code`` / ``hint`` / ``allowed``."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        hint: str = "",
        allowed: Any = None,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.allowed = allowed
        self.status = status
        self.retry_after = retry_after

    def __str__(self) -> str:
        parts = [f"{self.code}: {self.message}"]
        if self.hint:
            parts.append(f"hint={self.hint}")
        if self.allowed not in (None, {}):
            parts.append(f"allowed={self.allowed}")
        return " | ".join(parts)


class MarketUnavailable(RuntimeError):
    """Network/DNS/timeout failure; safe to retry after a short sleep."""


def _error_from_response(response: httpx.Response) -> MarketError:
    code = f"HTTP_{response.status_code}"
    message = response.text[:500]
    hint = ""
    allowed: Any = None
    try:
        body = response.json()
        error = body.get("error", body) if isinstance(body, dict) else {}
        if isinstance(error, dict):
            code = str(error.get("code") or code)
            message = str(error.get("message") or message)
            hint = str(error.get("hint") or "")
            allowed = error.get("allowed")
    except ValueError:
        pass

    retry_after: float | None = None
    if isinstance(allowed, dict):
        raw_retry = allowed.get("retry_after_s")
        if raw_retry is not None:
            try:
                retry_after = float(raw_retry)
            except (TypeError, ValueError):
                pass
    if retry_after is None:
        header = response.headers.get("Retry-After")
        if header:
            try:
                retry_after = float(header)
            except (TypeError, ValueError):
                pass

    return MarketError(
        code,
        message,
        hint=hint,
        allowed=allowed,
        status=response.status_code,
        retry_after=retry_after,
    )


class MarketClient:
    """Synchronous API client used by the polling agent loop."""

    MIN_INTERVAL_S = 0.22  # < 5 requests/second

    def __init__(
        self,
        base_url: str,
        team_key: str,
        *,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.team_key = team_key
        self._last_request = 0.0
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {team_key}"},
            timeout=timeout,
            transport=transport,
        )

    # ------------------------------------------------------------------ core
    def _throttle(self) -> None:
        wait = self.MIN_INTERVAL_S - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        self._throttle()
        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise MarketUnavailable(f"{type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            raise _error_from_response(response)
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise MarketError(
                "BAD_RESPONSE",
                f"Expected JSON, got: {response.text[:200]}",
                status=response.status_code,
            ) from exc

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -------------------------------------------------------------- endpoints
    def version(self) -> dict[str, Any]:
        return self._request("GET", "/v1/version")

    def state(self) -> dict[str, Any]:
        return self._request("GET", "/v1/state")

    def calendar(self) -> dict[str, Any]:
        return self._request("GET", "/v1/calendar")

    def appraisals(self) -> dict[str, Any]:
        return self._request("GET", "/v1/appraisals")

    def submit_appraisal(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/v1/appraisals", json={"items": items})

    def offers(self) -> dict[str, Any]:
        return self._request("GET", "/v1/offers")

    def submit_offer(self, offer: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/offers", json=offer)

    def submit_offers(
        self,
        offers: list[dict[str, Any]],
        *,
        replace_existing: bool = True,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/offers:batch",
            json={"offers": offers, "replace_existing": replace_existing},
        )

    def update_offer(self, offer_id: str, **changes: Any) -> dict[str, Any]:
        return self._request("PUT", f"/v1/offers/{offer_id}", json=changes)

    def cancel_offer(self, offer_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/offers/{offer_id}")

    def results(self, round_id: str | None = None) -> dict[str, Any]:
        path = "/v1/results"
        if round_id:
            path += f"?round={round_id}"
        return self._request("GET", path)

    def standings(self) -> dict[str, Any]:
        return self._request("GET", "/v1/standings")

    def leaderboard(self) -> dict[str, Any]:
        return self._request("GET", "/v1/leaderboard")

    def profile(self) -> dict[str, Any]:
        return self._request("GET", "/v1/profile")

    def set_profile(self, team_name: str, university: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/profile",
            json={"team_name": team_name, "university": university},
        )

    def reveal(self) -> dict[str, Any]:
        """Sandbox only, after CLOSED.  Competition markets return NOT_AVAILABLE."""
        return self._request("GET", "/v1/sandbox/reveal")

    def reset(self) -> dict[str, Any]:
        """Sandbox only: start a new practice game."""
        return self._request("POST", "/v1/sandbox/reset")
