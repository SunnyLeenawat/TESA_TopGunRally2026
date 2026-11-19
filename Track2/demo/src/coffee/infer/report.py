"""Day 1–3 deliverables from instance detections. No PyTorch."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from coffee.config import AppConfig, ConfigError
from coffee.infer.ncnn_seg import Detection

logger = logging.getLogger("coffee.infer.report")

_COLORS = (
    (70, 170, 50),
    (40, 140, 230),
    (220, 220, 40),
    (40, 90, 180),
    (40, 40, 210),
    (200, 190, 70),
    (180, 80, 180),
)


@dataclass(frozen=True)
class Bean:
    class_id: int
    class_name: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    anomaly: float | None
    mask: np.ndarray


@dataclass(frozen=True)
class ImageResult:
    image: str
    class_id: int
    class_name: str
    beans: tuple[Bean, ...]
    latency_ms: float


def list_inputs(path: Path, extensions: tuple[str, ...], class_names: tuple[str, ...]) -> list[Path]:
    """A file, a flat folder, or one level of class folders."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ConfigError(f"Input not found: {path}")
    known = set(class_names)
    found: list[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix.lower() in extensions:
            found.append(child)
        elif child.is_dir() and child.name in known:
            found.extend(
                sorted(
                    item
                    for item in child.iterdir()
                    if item.is_file() and item.suffix.lower() in extensions
                )
            )
    if not found:
        raise ConfigError(f"No images in {path}.")
    return found


def vote(beans: tuple[Bean, ...] | list[Bean]) -> tuple[int, str]:
    """Image class is the majority bean class. Ties go to the higher score sum, then the lower id."""
    if not beans:
        return -1, "none"
    counts: dict[int, int] = {}
    scores: dict[int, float] = {}
    names: dict[int, str] = {}
    for bean in beans:
        counts[bean.class_id] = counts.get(bean.class_id, 0) + 1
        scores[bean.class_id] = scores.get(bean.class_id, 0.0) + bean.score
        names[bean.class_id] = bean.class_name
    chosen = max(counts, key=lambda class_id: (counts[class_id], scores[class_id], -class_id))
    return chosen, names[chosen]


def count_classes(beans: tuple[Bean, ...] | list[Bean], class_names: tuple[str, ...]) -> dict[str, int]:
    counts = {name: 0 for name in class_names}
    for bean in beans:
        counts[bean.class_name] += 1
    return counts


def defect_ratio(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return (total - counts.get("good", 0)) / total


def draw_beans(image: np.ndarray, beans: tuple[Bean, ...] | list[Bean]) -> np.ndarray:
    canvas = image.copy()
    if not beans:
        return canvas
    tint = canvas.copy()
    covered = np.zeros(canvas.shape[:2], dtype=bool)
    for bean in beans:
        color = _COLORS[bean.class_id % len(_COLORS)]
        region = bean.mask > 0
        if region.shape != covered.shape:
            region = cv2.resize(bean.mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        tint[region] = color
        covered |= region
    blended = cv2.addWeighted(tint, 0.45, canvas, 0.55, 0)
    canvas[covered] = blended[covered]
    for bean in beans:
        color = _COLORS[bean.class_id % len(_COLORS)]
        x1, y1, x2, y2 = (int(round(value)) for value in bean.box_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{bean.class_name} {bean.score:.2f}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def run_day(
    cfg: AppConfig,
    images: list[Path],
    destination: Path,
    predict,
    input_root: Path | None = None,
) -> list[ImageResult]:
    destination.mkdir(parents=True, exist_ok=True)
    results: list[ImageResult] = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise ConfigError(f"Could not read {path}.")
        started = time.perf_counter()
        detections = list(predict(image))
        detections, anomalies = _maybe_fuse(cfg, image, detections)
        latency_ms = (time.perf_counter() - started) * 1000
        beans = tuple(_bean(cfg.classes, detection, anomaly) for detection, anomaly in zip(detections, anomalies))
        class_id, class_name = vote(beans)
        name = _output_name(path, input_root)
        result = ImageResult(name, class_id, class_name, beans, latency_ms)
        results.append(result)
        if cfg.day in (2, 3):
            overlay_path = destination / "overlays" / Path(name).with_suffix(".jpg").name
            _write_image(overlay_path, draw_beans(image, beans))
        counts = count_classes(beans, cfg.classes)
        logger.info(
            "%s class=%s beans=%s defect_ratio=%.3f latency_ms=%.1f",
            name,
            class_name,
            sum(counts.values()),
            defect_ratio(counts),
            latency_ms,
        )
    if cfg.day == 1:
        _write_csv(destination / "predictions.csv", results)
    if cfg.day == 3:
        _write_json(destination / "predictions.json", cfg, results)
    return results


def _maybe_fuse(cfg: AppConfig, image: np.ndarray, detections: list[Detection]) -> tuple[list[Detection], list[float | None]]:
    if not cfg.ae_enabled or not cfg.ae_fusion_enabled:
        return detections, [None] * len(detections)
    from coffee.ae.bank import load_bank
    from coffee.ae.crops import masked_square
    from coffee.ae.fuse import fuse
    from coffee.infer.ncnn_encoder import NcnnEncoder

    encoder = NcnnEncoder(cfg.model_dir / "ae_encoder_ncnn", cfg.ncnn_threads, cfg.ncnn_use_vulkan)
    bank = load_bank(cfg.model_dir / "ae_bank.npz")
    updated: list[Detection] = []
    anomalies: list[float | None] = []
    for detection in detections:
        square = masked_square(image, detection.mask, detection.box_xyxy, cfg.ae_crop_size)
        decision = fuse(
            detection.class_id,
            detection.score,
            encoder.embed(square),
            bank,
            enabled=True,
            k=cfg.ae_knn_k,
            knn_share=cfg.ae_knn_share,
            low_conf=cfg.ae_low_conf,
            confusable=cfg.ae_confusable,
        )
        updated.append(
            Detection(
                class_id=decision.class_id,
                score=decision.score,
                box_xyxy=detection.box_xyxy,
                mask=detection.mask,
                input_mask=detection.input_mask,
            )
        )
        anomalies.append(decision.anomaly)
    return updated, anomalies


def _bean(class_names: tuple[str, ...], detection: Detection, anomaly: float | None) -> Bean:
    if detection.class_id < 0 or detection.class_id >= len(class_names):
        raise ConfigError(f"Detection class id {detection.class_id} is outside the configured classes.")
    box = tuple(float(value) for value in detection.box_xyxy[:4])
    return Bean(detection.class_id, class_names[detection.class_id], float(detection.score), box, anomaly, detection.mask)


def _write_csv(path: Path, results: list[ImageResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "class_id", "class_name", "beans", "latency_ms"])
        for result in results:
            writer.writerow(
                [
                    result.image,
                    result.class_id,
                    result.class_name,
                    len(result.beans),
                    f"{result.latency_ms:.1f}",
                ]
            )
    logger.info("Wrote %s", path)


def _write_json(path: Path, cfg: AppConfig, results: list[ImageResult]) -> None:
    payload = {
        "day": cfg.day,
        "images": [_image_payload(cfg.classes, result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


def _image_payload(class_names: tuple[str, ...], result: ImageResult) -> dict:
    counts = count_classes(result.beans, class_names)
    return {
        "image": result.image,
        "latency_ms": round(result.latency_ms, 1),
        "total": sum(counts.values()),
        "defect_ratio": round(defect_ratio(counts), 6),
        "counts": counts,
        "per_bean": [
            {
                "class_id": bean.class_id,
                "class_name": bean.class_name,
                "score": round(bean.score, 4),
                "box_xyxy": [round(value, 2) for value in bean.box_xyxy],
                "anomaly": None if bean.anomaly is None else round(bean.anomaly, 4),
            }
            for bean in result.beans
        ],
    }


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ConfigError(f"Could not write {path}.")


def _output_name(path: Path, root: Path | None) -> str:
    if root is not None and path.parent != root:
        return f"{path.parent.name}__{path.name}"
    return path.name

