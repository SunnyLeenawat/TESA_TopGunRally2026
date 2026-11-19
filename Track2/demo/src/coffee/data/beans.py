"""Bean instances: mask, polygon, and crop."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Instance:
    class_name: str
    polygon_xy: np.ndarray  # float32 Nx2 in full-image pixels
    crop_bgr: np.ndarray
    crop_mask: np.ndarray  # uint8, 0 or 255, same size as crop_bgr

    @property
    def center(self) -> tuple[float, float]:
        points = self.polygon_xy
        return float(points[:, 0].mean()), float(points[:, 1].mean())


def mask_to_instance(
    image: np.ndarray,
    mask: np.ndarray,
    class_name: str,
    *,
    min_area_frac: float,
    max_area_frac: float,
    max_aspect: float,
) -> Instance | None:
    """Keep one foreground mask when its size and shape look like a single bean."""
    binary = (mask > 0).astype(np.uint8)
    if binary.shape[:2] != image.shape[:2]:
        binary = cv2.resize(binary, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    height, width = binary.shape
    area = int(binary.sum())
    image_area = height * width
    if area < min_area_frac * image_area or area > max_area_frac * image_area:
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    _x, _y, box_w, box_h = cv2.boundingRect(contour)
    if box_w < 2 or box_h < 2:
        return None
    if max(box_w, box_h) / float(min(box_w, box_h)) > max_aspect:
        return None
    if _touches_border(binary):
        return None
    polygon = _simplify(contour)
    if polygon.shape[0] < 3:
        return None
    crop_bgr, crop_mask = _crop(image, binary, contour)
    if crop_bgr.size == 0:
        return None
    return Instance(class_name=class_name, polygon_xy=polygon, crop_bgr=crop_bgr, crop_mask=crop_mask)


def nms(instances: list[Instance], iou_threshold: float = 0.45) -> list[Instance]:
    """Drop a smaller instance when its box heavily overlaps a larger one."""
    ordered = sorted(instances, key=lambda item: int(item.crop_mask.sum()), reverse=True)
    kept: list[Instance] = []
    for candidate in ordered:
        if any(_box_iou(_bbox(candidate), _bbox(other)) > iou_threshold for other in kept):
            continue
        kept.append(candidate)
    return kept


def draw_overlay(image: np.ndarray, instances: list[Instance]) -> np.ndarray:
    canvas = image.copy()
    for instance in instances:
        points = np.round(instance.polygon_xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        x, y = instance.center
        cv2.putText(
            canvas,
            instance.class_name,
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _simplify(contour: np.ndarray) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, 0.008 * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.reshape(-1, 2).astype(np.float32)


def _crop(image: np.ndarray, mask: np.ndarray, contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, box_w, box_h = cv2.boundingRect(contour)
    pad = 2
    height, width = mask.shape
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + box_w + pad)
    y1 = min(height, y + box_h + pad)
    return image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy() * 255


def _touches_border(mask: np.ndarray) -> bool:
    band = 2
    border = np.concatenate(
        [
            mask[:band].ravel(),
            mask[-band:].ravel(),
            mask[:, :band].ravel(),
            mask[:, -band:].ravel(),
        ]
    )
    return bool(border.any())


def _bbox(instance: Instance) -> tuple[float, float, float, float]:
    points = instance.polygon_xy
    return float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
