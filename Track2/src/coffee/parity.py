"""Compare Ultralytics PyTorch detections with the pure-NCNN decoder on the same images."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from coffee.config import AppConfig, ConfigError, load_config
from coffee.infer.ncnn_seg import Detection, NcnnSegmenter, ncnn_model_dir

logger = logging.getLogger("coffee.parity")

_BOX_IOU = 0.5
_MASK_IOU = 0.5
_SCORE_TOLERANCE = 0.05


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    try:
        cfg = load_config()
        images = _sample_images(cfg)
        report = compare_images(cfg, images)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    for line in report.lines:
        logger.info("%s", line)
    logger.info(
        "Parity %s. Mean NCNN latency %.1f ms over %s images.",
        "passed" if report.ok else "failed",
        report.mean_latency_ms,
        len(images),
    )
    return 0 if report.ok else 1


class Report:
    def __init__(self) -> None:
        self.ok = True
        self.lines: list[str] = []
        self.latencies_ms: list[float] = []

    @property
    def mean_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.mean(self.latencies_ms))


def compare_images(cfg: AppConfig, images: list[Path]) -> Report:
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import process_mask, scale_boxes

    weights = cfg.output_dir / "runs" / "yolo26n-seg" / "weights" / "best.pt"
    if not weights.is_file():
        raise ConfigError(f"Missing {weights}. Run train-yolo first.")
    segmenter = NcnnSegmenter(
        ncnn_model_dir(cfg.model_dir, cfg.output_dir),
        cfg.classes,
        cfg.imgsz,
        cfg.ncnn_threads,
        cfg.yolo_conf,
        cfg.yolo_iou,
        cfg.ncnn_use_vulkan,
    )
    model = YOLO(str(weights))
    report = Report()
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ConfigError(f"Could not read {image_path}.")
        lettered = LetterBox(cfg.imgsz)(image=image)
        tensor = torch.from_numpy(_torch_input(lettered))[None]
        with torch.no_grad():
            prediction, prototypes = model.model(tensor)[0]
        kept = non_max_suppression(
            prediction,
            cfg.yolo_conf,
            cfg.yolo_iou,
            nc=len(cfg.classes),
            max_det=300,
        )[0]
        reference = _reference_detections(kept, prototypes[0], lettered.shape[:2], image.shape[:2], scale_boxes, process_mask)
        started = time.perf_counter()
        ours = segmenter.predict(image)
        report.latencies_ms.append((time.perf_counter() - started) * 1000)
        problems = _match(reference, ours)
        status = "ok" if not problems else "FAIL"
        if problems:
            report.ok = False
        report.lines.append(
            f"{status} {image_path.name}: pytorch {len(reference)} ncnn {len(ours)}"
            + ("" if not problems else " " + "; ".join(problems))
        )
    return report


def _reference_detections(kept, prototypes, input_hw, original_hw, scale_boxes, process_mask) -> list[Detection]:
    if kept.shape[0] == 0:
        return []
    masks = process_mask(prototypes, kept[:, 6:], kept[:, :4], input_hw, upsample=True).cpu().numpy()
    boxes = scale_boxes(input_hw, kept[:, :4].clone(), original_hw).cpu().numpy()
    detections: list[Detection] = []
    for index in range(kept.shape[0]):
        detections.append(
            Detection(
                class_id=int(kept[index, 5].item()),
                score=float(kept[index, 4].item()),
                box_xyxy=boxes[index],
                mask=masks[index],
                input_mask=masks[index],
            )
        )
    return detections


def _match(reference: list[Detection], ours: list[Detection]) -> list[str]:
    used: set[int] = set()
    problems: list[str] = []
    for ref in reference:
        partner = _best_partner(ref, ours, used)
        if partner is None:
            problems.append(f"missing class {ref.class_id} score {ref.score:.3f}")
            continue
        used.add(partner)
        det = ours[partner]
        if abs(det.score - ref.score) > _SCORE_TOLERANCE:
            problems.append(f"score {det.score:.3f} vs {ref.score:.3f}")
        if _box_iou(det.box_xyxy, ref.box_xyxy) < _BOX_IOU:
            problems.append(f"box iou low for class {ref.class_id}")
        mask_iou = _mask_iou(det.input_mask, ref.input_mask)
        if mask_iou < _MASK_IOU:
            problems.append(f"mask iou {mask_iou:.2f} for class {ref.class_id}")
    extras = len(ours) - len(used)
    if extras:
        problems.append(f"{extras} extra ncnn detections")
    return problems


def _best_partner(ref: Detection, ours: list[Detection], used: set[int]) -> int | None:
    best_index = None
    best_iou = _BOX_IOU
    for index, det in enumerate(ours):
        if index in used or det.class_id != ref.class_id:
            continue
        iou = _box_iou(ref.box_xyxy, det.box_xyxy)
        if iou >= best_iou:
            best_iou = iou
            best_index = index
    return best_index


def _sample_images(cfg: AppConfig) -> list[Path]:
    folders = [
        cfg.output_dir / "yolo_bundle" / "images" / "val",
        cfg.output_dir / "synth" / "images",
    ]
    found: list[Path] = []
    for folder in folders:
        if folder.is_dir():
            found.extend(sorted(folder.glob("*.jpg"))[:3])
    if not found:
        raise ConfigError("No images under the val split or synth folder to check parity.")
    return found


def _torch_input(image_bgr: np.ndarray) -> np.ndarray:
    rgb = image_bgr[:, :, ::-1]
    return np.ascontiguousarray(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    left = a > 0
    right = b > 0
    if left.shape != right.shape:
        right_image = (right.astype(np.uint8) * 255)
        right = cv2.resize(right_image, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


if __name__ == "__main__":
    raise SystemExit(main())
