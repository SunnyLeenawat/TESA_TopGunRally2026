"""Day 3 inference writer. One instance list feeds the mask, polygons, counts, and JSON."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from coffee.config import AppConfig, ConfigError
from coffee.data.roi import crop_image, estimate_roi
from coffee.eval.validate import ContractError, validate_outputs
from coffee.infer.instances import (
    FinalInstance,
    counts_for,
    defect_ratio,
    paper_gate,
    paste_crop_mask,
    resolve_instances,
)
from coffee.infer.ncnn_seg import Detection

logger = logging.getLogger("coffee.infer")

_PALETTE: tuple[tuple[int, int, int], ...] = (
    (40, 80, 180),
    (0, 200, 255),
    (60, 180, 60),
    (220, 180, 40),
    (180, 80, 200),
    (30, 30, 30),
    (40, 40, 220),
)


def run_infer(
    cfg: AppConfig,
    input_path: Path,
    weights: Path,
    out_dir: Path,
    predict: Callable[[np.ndarray], list[Detection]] | None = None,
    *,
    conf: float | None = None,
    dedup_iou: float | None = None,
    refine: bool | None = None,
    write_vis: bool = True,
) -> bool:
    """Write the contract for every readable image. Return False when validation fails."""
    images = list_images(input_path, cfg.image_extensions)
    chosen_conf = cfg.day3_infer_conf if conf is None else conf
    chosen_dedup = cfg.day3_dedup_iou if dedup_iou is None else dedup_iou
    chosen_refine = cfg.day3_refine if refine is None else refine
    if predict is None:
        predict = _ncnn_predict(cfg, weights, chosen_conf)
    logger.info("Infer conf=%s dedup=%s refine=%s", chosen_conf, chosen_dedup, chosen_refine)
    for folder in ("seg_mask", "mqtt", "vis"):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    stages: list[dict[str, object]] = []
    for image_path in images:
        record, stage = _one_image(
            cfg, image_path, out_dir, predict, chosen_dedup, chosen_refine, write_vis
        )
        if record is None:
            continue
        records.append(record)
        stages.append(stage)
    prediction_started = time.perf_counter()
    (out_dir / "predictions.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (out_dir / "stages.json").write_text(json.dumps(stages, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote predictions.json in %.0f ms", (time.perf_counter() - prediction_started) * 1000)
    written = [path for path in images if any(item["image_name"] == path.name for item in records)]
    try:
        validate_outputs(out_dir, written, cfg.submission_names, cfg.good_name)
    except ContractError as exc:
        for problem in exc.problems:
            logger.error("%s", problem)
        return False
    logger.info("Output contract passed for %s image(s).", len(written))
    return True


def list_images(path: Path, extensions: tuple[str, ...]) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ConfigError(f"Input not found: {path}")
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in extensions)


def _one_image(
    cfg: AppConfig,
    image_path: Path,
    out_dir: Path,
    predict: Callable[[np.ndarray], list[Detection]],
    dedup_iou: float,
    refine: bool,
    write_vis: bool,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    started = time.perf_counter()
    decode_started = started
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    decode_ms = (time.perf_counter() - decode_started) * 1000
    if image is None:
        logger.error("Could not read %s. No valid empty result is possible without a size.", image_path)
        return None, None
    try:
        roi_started = time.perf_counter()
        roi = estimate_roi(
            image,
            dark_delta=cfg.day3_roi_dark_delta,
            margin_frac=cfg.day3_roi_margin_frac,
            fallback_frac=cfg.day3_roi_fallback_frac,
        )
        crop = crop_image(image, roi)
        roi_ms = (time.perf_counter() - roi_started) * 1000
        ncnn_started = time.perf_counter()
        detections = predict(crop)
        ncnn_ms = (time.perf_counter() - ncnn_started) * 1000
        post_started = time.perf_counter()
        candidates = []
        for detection in detections:
            if detection.class_id < 0 or detection.class_id >= len(cfg.classes):
                logger.error("%s dropped class id %s", image_path.name, detection.class_id)
                continue
            full_mask = paste_crop_mask(detection.mask, image.shape[:2], roi)
            if refine:
                full_mask = paper_gate(image, full_mask, cfg.day3_roi_dark_delta)
            candidates.append((cfg.classes[detection.class_id], float(detection.score), full_mask))
        label, instances = resolve_instances(
            candidates,
            image.shape[:2],
            dedup_iou=dedup_iou,
            min_area=cfg.day3_min_area,
        )
        post_ms = (time.perf_counter() - post_started) * 1000
    except Exception:
        logger.exception("Inference failed for %s. Writing an empty valid result.", image_path.name)
        label = np.zeros(image.shape[:2], np.uint8)
        instances = []
        roi_ms = ncnn_ms = post_ms = 0.0
    mask_ms = _write_mask(out_dir, image_path.name, label)
    json_started = time.perf_counter()
    total_ms = int(round((time.perf_counter() - started) * 1000))
    _write_mqtt(out_dir, image_path.name, instances, cfg, total_ms)
    json_ms = (time.perf_counter() - json_started) * 1000
    vis_started = time.perf_counter()
    vis_path = out_dir / "vis" / f"{image_path.stem}.png"
    if write_vis:
        _write_vis(vis_path, image, label, instances, cfg)
    elif not cv2.imwrite(str(vis_path), np.zeros((8, 8, 3), np.uint8)):
        raise ConfigError(f"Could not write {vis_path}.")
    vis_ms = (time.perf_counter() - vis_started) * 1000
    stage = {
        "image": image_path.name,
        "decode_ms": round(decode_ms, 1),
        "roi_ms": round(roi_ms, 1),
        "ncnn_ms": round(ncnn_ms, 1),
        "post_ms": round(post_ms, 1),
        "mask_ms": round(mask_ms, 1),
        "json_ms": round(json_ms, 1),
        "vis_ms": round(vis_ms, 1),
        "latency_ms": total_ms,
        "beans": len(instances),
    }
    logger.info(
        "%s decode=%.0f roi=%.0f ncnn=%.0f post=%.0f mask=%.0f json=%.0f vis=%.0f total=%s beans=%s",
        image_path.name,
        decode_ms,
        roi_ms,
        ncnn_ms,
        post_ms,
        mask_ms,
        json_ms,
        vis_ms,
        total_ms,
        len(instances),
    )
    return _prediction(image_path.name, instances), stage


def _write_mask(out_dir: Path, image_name: str, label: np.ndarray) -> float:
    started = time.perf_counter()
    path = out_dir / "seg_mask" / f"{Path(image_name).stem}.png"
    if not cv2.imwrite(str(path), label, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise ConfigError(f"Could not write {path}.")
    return (time.perf_counter() - started) * 1000


def _prediction(image_name: str, instances: list[FinalInstance]) -> dict[str, object]:
    return {
        "image_name": image_name,
        "objects": [
            {"id": instance.id, "class": instance.class_name, "polygon": instance.polygon}
            for instance in instances
        ],
    }


def _write_mqtt(
    out_dir: Path,
    image_name: str,
    instances: list[FinalInstance],
    cfg: AppConfig,
    latency_ms: int,
) -> None:
    counts = counts_for(instances, cfg.submission_names)
    payload = {
        "image": image_name,
        "latency_ms": int(latency_ms),
        "counts": counts,
        "defect_ratio": defect_ratio(counts, cfg.good_name),
    }
    path = out_dir / "mqtt" / f"{Path(image_name).stem}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_vis(
    path: Path,
    image: np.ndarray,
    label: np.ndarray,
    instances: list[FinalInstance],
    cfg: AppConfig,
) -> None:
    canvas = image.copy()
    for instance in instances:
        color = _PALETTE[cfg.class_to_id[instance.class_name] % len(_PALETTE)]
        pixels = label == instance.id
        canvas[pixels] = (0.55 * canvas[pixels] + 0.45 * np.array(color)).astype(np.uint8)
        contour = np.array(instance.polygon, np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [contour], True, color, 2, cv2.LINE_AA)
        x, y = instance.polygon[0]
        cv2.putText(canvas, instance.class_name, (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise ConfigError(f"Could not write {path}.")


def _ncnn_predict(cfg: AppConfig, weights: Path, conf: float) -> Callable[[np.ndarray], list[Detection]]:
    from coffee.infer.ncnn_seg import NcnnSegmenter

    segmenter = NcnnSegmenter(
        weights,
        cfg.classes,
        cfg.imgsz,
        cfg.ncnn_threads,
        conf,
        cfg.yolo_iou,
        cfg.ncnn_use_vulkan,
    )

    def predict(image: np.ndarray) -> list[Detection]:
        return segmenter.predict(image, conf=conf)

    return predict
