from __future__ import annotations

import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

from coffee.client.http_api import HttpResponse, HttpStop, fetch_image, retry, submit_predictions
from coffee.client.mqtt import resolve_topic
from coffee.client.run import dispatch, run_batch
from coffee.config import ConfigError
from coffee.mock.check import check_tree
from coffee.mock.server import _handler, _parts


def test_fetch_image_rejects_a_path(tmp_path: Path) -> None:
    def get(url: str, timeout: float) -> HttpResponse:
        assert url.endswith("?mode=train")
        return HttpResponse(200, b"jpeg", headers={"X-Image-ID": "img_1", "X-Image-Name": "../secret.jpg"})

    with pytest.raises(ConfigError, match="plain file name"):
        fetch_image(get, "http://mock", "/api/v1/image", "train", tmp_path, 1)


def test_fetch_image_writes_the_header_name(tmp_path: Path) -> None:
    def get(url: str, timeout: float) -> HttpResponse:
        return HttpResponse(200, b"jpeg-bytes", headers={"X-Image-ID": "img_9", "X-Image-Name": "bean.jpg"})

    img_id, name = fetch_image(get, "http://mock", "/api/v1/image", "train", tmp_path, 1)
    assert (img_id, name) == ("img_9", "bean.jpg")
    assert (tmp_path / "bean.jpg").read_bytes() == b"jpeg-bytes"


def test_submit_409_is_not_retried() -> None:
    calls = {"n": 0}

    def post(url: str, payload: dict, timeout: float) -> HttpResponse:
        calls["n"] += 1
        assert payload["img_id"] == "img_9"
        assert payload["predictions"]["image_name"] == "bean.jpg"
        return HttpResponse(409, b"duplicate")

    with pytest.raises(HttpStop, match="already submitted"):
        retry(
            lambda: submit_predictions(post, "http://mock", "/api/v1/submit", "img_9", "bean.jpg", [], 1),
            3,
            "submit",
        )
    assert calls["n"] == 1


def test_resolve_topic_fills_placeholders() -> None:
    from coffee.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    placeholder = cfg.__class__(**{**cfg.__dict__, "mqtt_topic": "/event/abc/TRACK/EVENT"})
    assert resolve_topic(placeholder) == f"/event/abc/{cfg.mqtt_track}/{cfg.mqtt_event}"
    finished = cfg.__class__(**{**cfg.__dict__, "mqtt_topic": "/event/abc/EdgeAI/system-testing"})
    assert resolve_topic(finished) == "/event/abc/EdgeAI/system-testing"


def test_run_batch_posts_json_for_one_image_then_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from coffee.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "http_base": "https://example.test",
            "http_token": "token",
            "http_images_path": "/api/v1/image",
            "http_submit_path": "/api/v1/submit",
            "http_mode": "train",
            "http_retries": 1,
            "mqtt_retries": 1,
            "mqtt_password": "key",
            "mqtt_topic": "/event/abc/EdgeAI/system-testing",
        }
    )
    state = {"gets": 0}
    posted: list[dict] = []
    published: list[bytes] = []

    def get(url: str, timeout: float) -> HttpResponse:
        state["gets"] += 1
        if state["gets"] == 1:
            return HttpResponse(
                200,
                b"image",
                headers={"X-Image-ID": "img_1", "X-Image-Name": "bean.jpg", "X-Mode": "train"},
            )
        return HttpResponse(403, b"closed")

    def post(url: str, payload: dict, timeout: float) -> HttpResponse:
        posted.append(payload)
        return HttpResponse(200, b"{}")

    def infer(cfg, image_dir: Path, weights: Path, contract: Path) -> bool:
        (contract / "mqtt").mkdir(parents=True)
        (contract / "predictions.json").write_text(
            json.dumps([{"image_name": "bean.jpg", "objects": [{"id": 1, "class": "good", "polygon": [[0, 0]]}]}]),
            encoding="utf-8",
        )
        (contract / "mqtt" / "bean.json").write_text(
            json.dumps({"image": "bean.jpg", "latency_ms": 4, "counts": {}, "defect_ratio": 0.0}),
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr("coffee.client.run.run_infer", infer)

    class _Publisher:
        def __init__(self, config) -> None:
            self.config = config

        def __enter__(self) -> _Publisher:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def publish(self, payload: bytes) -> None:
            published.append(payload)

    code = run_batch(cfg, tmp_path, tmp_path / "out", get, post, _Publisher, None)
    assert code == 0
    assert state["gets"] == 2
    assert posted[0]["img_id"] == "img_1"
    assert posted[0]["predictions"]["objects"][0]["class"] == "good"
    assert json.loads(published[0].decode("utf-8"))["image"] == "bean.jpg"


def test_retry_raises_the_last_error() -> None:
    calls = {"n": 0}

    def action() -> None:
        calls["n"] += 1
        raise ConfigError("nope")

    with pytest.raises(ConfigError, match="nope"):
        retry(action, 2, "probe")
    assert calls["n"] == 2


def test_upload_and_mqtt_overlap() -> None:
    order: list[str] = []
    lock = threading.Lock()

    def mark(name: str) -> None:
        with lock:
            order.append(name)
        time.sleep(0.05)
        with lock:
            order.append(name + "-end")

    dispatch(lambda: mark("http"), lambda: mark("mqtt"))
    assert order.index("http") < order.index("mqtt-end")
    assert order.index("mqtt") < order.index("http-end")


def test_check_tree_accepts_a_matching_bundle(tmp_path: Path) -> None:
    _bundle(tmp_path, good=1)
    assert check_tree(tmp_path) == []


def test_check_tree_rejects_a_count_mismatch(tmp_path: Path) -> None:
    _bundle(tmp_path, good=1)
    message = json.loads((tmp_path / "mqtt" / "bean.json").read_text(encoding="utf-8"))
    message["counts"]["good"] = 0
    (tmp_path / "mqtt" / "bean.json").write_text(json.dumps(message), encoding="utf-8")
    problems = check_tree(tmp_path)
    assert any("counts" in problem for problem in problems)


def test_mock_serves_an_image_and_checks_a_post(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    cv2.imwrite(str(image_dir / "bean.jpg"), np.full((20, 20, 3), 180, np.uint8))
    report = tmp_path / "report.json"
    bundle = tmp_path / "bundle"
    _bundle(bundle, good=0)
    archive = tmp_path / "team_55_Day3.zip"
    _zip(bundle, archive)
    predictions = (bundle / "predictions.json").read_bytes()
    handler = _handler(image_dir, report)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/images", timeout=2) as response:
            assert json.loads(response.read().decode("utf-8")) == ["bean.jpg"]
        body, content_type = _multipart(predictions, archive.read_bytes())
        request = Request(
            f"http://127.0.0.1:{port}/submit",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert _parts(body, content_type)["results"][0] == "results.json"
    finally:
        server.shutdown()


def _bundle(root: Path, good: int) -> None:
    mask = np.zeros((12, 12), np.uint8)
    objects: list[dict] = []
    counts = {name: 0 for name in ("good", "shell", "black", "husk", "broken", "parchment", "immature")}
    if good:
        mask[2:8, 2:8] = 1
        objects.append({"id": 1, "class": "good", "polygon": [[2, 2], [8, 2], [8, 8], [2, 8]]})
        counts["good"] = 1
    (root / "seg_mask").mkdir(parents=True, exist_ok=True)
    (root / "mqtt").mkdir(exist_ok=True)
    cv2.imwrite(str(root / "seg_mask" / "bean.png"), mask)
    (root / "predictions.json").write_text(
        json.dumps([{"image_name": "bean.jpg", "objects": objects}]),
        encoding="utf-8",
    )
    (root / "mqtt" / "bean.json").write_text(
        json.dumps({"image": "bean.jpg", "latency_ms": 1, "counts": counts, "defect_ratio": 0.0}),
        encoding="utf-8",
    )


def _zip(root: Path, destination: Path) -> None:
    import zipfile

    with zipfile.ZipFile(destination, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())


def _multipart(results: bytes, archive: bytes) -> tuple[bytes, str]:
    boundary = "bound123"
    body = (
        f"--{boundary}\r\n".encode()
        + b'Content-Disposition: form-data; name="results"; filename="results.json"\r\n\r\n'
        + results
        + f"\r\n--{boundary}\r\n".encode()
        + b'Content-Disposition: form-data; name="archive"; filename="team_55_Day3.zip"\r\n\r\n'
        + archive
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"
