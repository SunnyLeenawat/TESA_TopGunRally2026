"""Fetch one image, infer it, then POST the JSON and publish MQTT together."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from coffee.client.http_api import HttpStop, fetch_image, requests_transport, retry, submit_predictions
from coffee.client.mqtt import Publisher
from coffee.config import AppConfig, ConfigError
from coffee.infer.contract import run_infer
from coffee.infer.instances import counts_for, defect_ratio

logger = logging.getLogger("coffee.client")


def run_client(cfg: AppConfig, weights: Path, out_dir: Path, limit: int | None = None) -> int:
    if not cfg.http_base:
        logger.error("COFFEE_HTTP_BASE is empty. Set it or run make day3-mock.")
        return 2
    if not cfg.http_token:
        logger.error("COFFEE_HTTP_TOKEN is empty. It must be the website password.")
        return 2
    if not cfg.mqtt_topic:
        logger.error("COFFEE_MQTT_TOPIC is empty. Set it before starting the client.")
        return 2
    if not cfg.mqtt_password:
        logger.error("COFFEE_MQTT_PASSWORD is empty. It must be the MQTT key, not the website password.")
        return 2
    if not weights.is_dir():
        logger.error("Weights path is not a folder: %s", weights)
        return 2
    if limit is not None and limit < 1:
        logger.error("--limit must be at least 1.")
        return 2
    get, post = requests_transport(cfg.http_token)
    return run_batch(cfg, weights, out_dir, get, post, Publisher, limit)


def run_batch(
    cfg: AppConfig,
    weights: Path,
    out_dir: Path,
    get: Callable,
    post: Callable,
    publisher_type: Callable,
    limit: int | None = None,
) -> int:
    """GET one image, infer it, then start the JSON POST and the MQTT publish together."""
    if limit is None:
        logger.info("HTTP mode %s. Fetching until the server returns 403.", cfg.http_mode)
    else:
        logger.info("HTTP mode %s. Fetching at most %s image(s).", cfg.http_mode, limit)
    sent = 0
    failed = False
    seen: set[str] = set()
    try:
        publisher_cm = publisher_type(cfg)
        publisher = publisher_cm.__enter__()
    except Exception as exc:
        logger.error("MQTT connect failed: %s", exc)
        return 2
    try:
        while limit is None or sent < limit:
            try:
                img_id, image_name = retry(
                    lambda: fetch_image(
                        get,
                        cfg.http_base,
                        cfg.http_images_path,
                        cfg.http_mode,
                        out_dir / "incoming",
                        cfg.http_timeout_s,
                    ),
                    cfg.http_retries,
                    "fetch image",
                )
            except HttpStop as exc:
                if exc.status == 403:
                    logger.info("Stopped fetching: %s", exc)
                    break
                logger.error("%s", exc)
                failed = True
                break
            except Exception as exc:
                logger.error("Could not fetch an image: %s", exc)
                failed = True
                break
            assert isinstance(img_id, str) and isinstance(image_name, str)
            if img_id in seen:
                logger.error("Server repeated img_id %s. Stopping so it is not submitted twice.", img_id)
                failed = True
                break
            seen.add(img_id)
            objects, payload = _infer_one(cfg, weights, out_dir, img_id, image_name)
            http_error, mqtt_error = dispatch(
                lambda img_id=img_id, image_name=image_name, objects=objects: _submit_one(
                    cfg, post, img_id, image_name, objects
                ),
                lambda payload=payload: retry(
                    lambda payload=payload: publisher.publish(payload),
                    cfg.mqtt_retries,
                    "mqtt publish",
                ),
            )
            sent += 1
            if http_error is not None:
                logger.error("Submit failed for %s: %s", img_id, http_error)
                failed = True
                if isinstance(http_error, HttpStop) and http_error.status in {401, 403}:
                    break
            if mqtt_error is not None:
                logger.error("MQTT failed for %s: %s", image_name, mqtt_error)
                failed = True
        if sent == 0 and not failed:
            logger.info("No image was fetched.")
            return 0
        if failed:
            return 1
        logger.info("Client finished. Submitted %s image(s).", sent)
        return 0
    finally:
        publisher_cm.__exit__(None, None, None)


def dispatch(upload: Callable[[], object], publish: Callable[[], object]) -> tuple[Exception | None, Exception | None]:
    """Start upload and publish together. One side does not wait to begin."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        upload_future = pool.submit(upload)
        publish_future = pool.submit(publish)
        upload_error = _result(upload_future)
        publish_error = _result(publish_future)
    return upload_error, publish_error


def _result(future) -> Exception | None:
    try:
        future.result()
    except Exception as exc:
        return exc
    return None


def _submit_one(
    cfg: AppConfig,
    post: Callable,
    img_id: str,
    image_name: str,
    objects: list[dict[str, object]],
) -> None:
    try:
        retry(
            lambda: submit_predictions(
                post,
                cfg.http_base,
                cfg.http_submit_path,
                img_id,
                image_name,
                objects,
                cfg.http_timeout_s,
            ),
            cfg.http_retries,
            f"submit {img_id}",
        )
    except HttpStop as exc:
        if exc.status == 409:
            logger.warning("%s", exc)
            return
        raise


def _infer_one(
    cfg: AppConfig,
    weights: Path,
    out_dir: Path,
    img_id: str,
    image_name: str,
) -> tuple[list[dict[str, object]], bytes]:
    source = out_dir / "incoming" / image_name
    round_dir = out_dir / "rounds" / _safe_id(img_id)
    image_dir = round_dir / "image"
    contract = round_dir / "contract"
    image_dir.mkdir(parents=True, exist_ok=True)
    target = image_dir / image_name
    if source.is_file():
        target.write_bytes(source.read_bytes())
    try:
        ok = run_infer(cfg, image_dir, weights, contract)
        if not ok:
            logger.error("Inference validation failed for %s. Submitting the written result.", image_name)
    except Exception:
        logger.exception("Inference failed for %s. Submitting an empty result.", image_name)
    return _round_payload(cfg, contract, image_name)


def _round_payload(
    cfg: AppConfig,
    contract: Path,
    image_name: str,
) -> tuple[list[dict[str, object]], bytes]:
    objects = _objects_for(contract / "predictions.json", image_name)
    message = _mqtt_message(cfg, contract, image_name)
    message["image"] = image_name
    return objects, (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


def _objects_for(path: Path, image_name: str) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return []
    chosen: list[object] = []
    for row in rows:
        if isinstance(row, dict) and row.get("image_name") == image_name:
            chosen = list(row.get("objects") or [])
            break
    else:
        if rows and isinstance(rows[0], dict):
            chosen = list(rows[0].get("objects") or [])
    cleaned: list[dict[str, object]] = []
    for item in chosen:
        if not isinstance(item, dict):
            continue
        cleaned.append({"id": item.get("id"), "class": item.get("class"), "polygon": item.get("polygon")})
    return cleaned


def _mqtt_message(cfg: AppConfig, contract: Path, image_name: str) -> dict[str, object]:
    path = contract / "mqtt" / f"{Path(image_name).stem}.json"
    if path.is_file():
        message = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(message, dict):
            return message
    counts = counts_for([], cfg.submission_names)
    return {
        "image": image_name,
        "latency_ms": 0,
        "counts": counts,
        "defect_ratio": defect_ratio(counts, cfg.good_name),
    }


def _safe_id(img_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in img_id)
    return cleaned or "image"
