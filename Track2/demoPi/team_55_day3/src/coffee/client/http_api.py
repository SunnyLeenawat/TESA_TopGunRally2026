"""One-image HTTP fetch and JSON submit. Callers pass the transport so tests do not open sockets."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from coffee.config import ConfigError

logger = logging.getLogger("coffee.client.http")


class HttpStop(ConfigError):
    """A status the caller must handle. retry() does not try again."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        super().__init__(detail)


class HttpResponse:
    def __init__(
        self,
        status: int,
        body: bytes,
        json_body: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.body = body
        self._json = json_body
        self.headers = {str(key).lower(): str(value) for key, value in (headers or {}).items()}

    def header(self, name: str) -> str:
        return self.headers.get(name.lower(), "").strip()

    def raise_for_status(self) -> None:
        if self.status < 200 or self.status >= 300:
            raise ConfigError(f"HTTP {self.status}: {self.body[:200]!r}")

    def json(self) -> object:
        if self._json is not None:
            return self._json
        import json

        return json.loads(self.body.decode("utf-8"))


def fetch_image(
    get: Callable[[str, float], HttpResponse],
    base: str,
    path: str,
    mode: str,
    destination: Path,
    timeout: float,
) -> tuple[str, str]:
    """GET one image. Return (img_id, image_name) and write the bytes under destination."""
    url = f"{base.rstrip('/')}{path}?mode={mode}"
    response = get(url, timeout)
    if response.status == 403:
        raise HttpStop(403, "Image feed closed or the test quota is spent.")
    if response.status == 401:
        raise HttpStop(401, "HTTP 401. COFFEE_HTTP_TOKEN is not the website password.")
    response.raise_for_status()
    img_id = response.header("X-Image-ID")
    image_name = response.header("X-Image-Name")
    if not img_id or not image_name:
        raise ConfigError("Image response is missing X-Image-ID or X-Image-Name.")
    safe = Path(image_name).name
    if safe != image_name or safe in {"", ".", ".."}:
        raise ConfigError(f"X-Image-Name is not a plain file name: {image_name!r}.")
    if not response.body:
        raise ConfigError(f"Empty body for {image_name}.")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / safe).write_bytes(response.body)
    logger.info("Fetched %s id %s mode %s", safe, img_id, response.header("X-Mode") or mode)
    return img_id, safe


def submit_predictions(
    post: Callable[[str, dict[str, object], float], HttpResponse],
    base: str,
    path: str,
    img_id: str,
    image_name: str,
    objects: list[dict[str, object]],
    timeout: float,
) -> None:
    """POST {img_id, predictions}. 409 means this id was already sent."""
    url = f"{base.rstrip('/')}{path}"
    payload = {"img_id": img_id, "predictions": {"image_name": image_name, "objects": objects}}
    response = post(url, payload, timeout)
    if response.status == 409:
        raise HttpStop(409, f"img_id {img_id} was already submitted.")
    if response.status == 401:
        raise HttpStop(401, "HTTP 401. COFFEE_HTTP_TOKEN is not the website password.")
    if response.status == 403:
        raise HttpStop(403, "Submit rejected with HTTP 403.")
    response.raise_for_status()


def retry(action: Callable[[], object], attempts: int, label: str) -> object:
    """Run action. Log and wait between failures. Do not retry HttpStop."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except HttpStop:
            raise
        except Exception as exc:
            last = exc
            logger.warning("%s attempt %s/%s failed: %s", label, attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
    assert last is not None
    raise last


def requests_transport(token: str) -> tuple[
    Callable[[str, float], HttpResponse],
    Callable[[str, dict[str, object], float], HttpResponse],
]:
    import requests

    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    def get(url: str, timeout: float) -> HttpResponse:
        response = session.get(url, timeout=timeout)
        return HttpResponse(response.status_code, response.content, headers=dict(response.headers))

    def post(url: str, payload: dict[str, object], timeout: float) -> HttpResponse:
        response = session.post(url, json=payload, timeout=timeout)
        return HttpResponse(response.status_code, response.content, headers=dict(response.headers))

    return get, post
