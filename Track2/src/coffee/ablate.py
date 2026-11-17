"""Score YOLO alone against YOLO plus the encoder bank on the prepared val images."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from coffee.ae.bank import load_bank
from coffee.ae.crops import masked_square
from coffee.ae.fuse import fuse
from coffee.config import AppConfig, ConfigError, load_config
from coffee.data.labels import read_yolo_seg
from coffee.infer.ncnn_encoder import NcnnEncoder
from coffee.infer.ncnn_seg import NcnnSegmenter, ncnn_model_dir

logger = logging.getLogger("coffee.ablate")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    try:
        cfg = load_config()
        report = run_ablation(cfg)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    logger.info(
        "Ablation on %s matched beans: YOLO %.3f, YOLO+AE %.3f, flips %s",
        report["matched"],
        report["yolo_accuracy"],
        report["fused_accuracy"],
        report["flips"],
    )
    return 0


def run_ablation(cfg: AppConfig) -> dict[str, float | int]:
    images = sorted((cfg.output_dir / "yolo_bundle" / "images" / "val").glob("*.jpg"))
    labels = cfg.output_dir / "yolo_bundle" / "labels" / "val"
    if not images:
        raise ConfigError("No val images for the ablation. Run prep and train-yolo first.")
    segmenter = NcnnSegmenter(
        ncnn_model_dir(cfg.model_dir, cfg.output_dir),
        cfg.classes,
        cfg.imgsz,
        cfg.ncnn_threads,
        cfg.yolo_conf,
        cfg.yolo_iou,
        cfg.ncnn_use_vulkan,
    )
    encoder = NcnnEncoder(cfg.model_dir / "ae_encoder_ncnn", cfg.ncnn_threads, cfg.ncnn_use_vulkan)
    bank = load_bank(cfg.model_dir / "ae_bank.npz")
    if bank.crop_size != cfg.ae_crop_size or bank.class_names != cfg.classes:
        raise ConfigError("Embedding bank does not match the current autoencoder config.")
    yolo_correct = 0
    fused_correct = 0
    matched = 0
    flips = 0
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ConfigError(f"Could not read {image_path}.")
        height, width = image.shape[:2]
        class_ids, polygons = read_yolo_seg(labels / f"{image_path.stem}.txt", width, height)
        truth = [(class_id, _bounds(polygon)) for class_id, polygon in zip(class_ids, polygons)]
        detections = segmenter.predict(image)
        used: set[int] = set()
        for detection in detections:
            partner = _match(detection.box_xyxy, truth, used)
            if partner is None:
                continue
            used.add(partner)
            matched += 1
            truth_id = truth[partner][0]
            if detection.class_id == truth_id:
                yolo_correct += 1
            try:
                square = masked_square(image, detection.mask, detection.box_xyxy, cfg.ae_crop_size)
            except ValueError:
                fused_id = detection.class_id
            else:
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
                fused_id = decision.class_id
                if decision.changed:
                    flips += 1
            if fused_id == truth_id:
                fused_correct += 1
    if matched == 0:
        raise ConfigError("Ablation matched no detections to labels.")
    report: dict[str, float | int] = {
        "images": len(images),
        "matched": matched,
        "yolo_accuracy": yolo_correct / matched,
        "fused_accuracy": fused_correct / matched,
        "flips": flips,
        "fusion_enabled": cfg.ae_fusion_enabled,
    }
    path = cfg.output_dir / "ae" / "ablation.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    logger.info("Wrote %s", path)
    return report


def _bounds(polygon: np.ndarray) -> np.ndarray:
    return np.array(
        [polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()],
        dtype=np.float32,
    )


def _match(box: np.ndarray, truth: list[tuple[int, np.ndarray]], used: set[int]) -> int | None:
    best = None
    best_iou = 0.5
    for index, (_class_id, other) in enumerate(truth):
        if index in used:
            continue
        iou = _iou(box, other)
        if iou >= best_iou:
            best_iou = iou
            best = index
    return best


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
