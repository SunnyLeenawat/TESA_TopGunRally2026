"""Turn detections into one instance list, then derive every output from it.

Overlap keeps the higher score. A bean that loses all of its pixels is dropped
from the mask, the polygons, and the counts together.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from coffee.data.roi import Roi


@dataclass(frozen=True)
class FinalInstance:
    id: int
    class_name: str
    score: float
    polygon: list[list[int]]


def paper_gate(image: np.ndarray, mask: np.ndarray, dark_delta: float = 12.0) -> np.ndarray:
    """Keep mask pixels that sit on a slightly dilated darker-than-paper foreground."""
    height, width = image.shape[:2]
    scale = min(1.0, 768.0 / float(max(height, width)))
    small = cv2.resize(image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))))
    lightness = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    sigma = max(3.0, 0.03 * max(small.shape[:2]))
    paper = cv2.GaussianBlur(lightness, (0, 0), sigma)
    foreground = ((paper - lightness) > dark_delta).astype(np.uint8) * 255
    foreground = cv2.dilate(foreground, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    full = cv2.resize(foreground, (width, height), interpolation=cv2.INTER_NEAREST)
    gated = mask.copy()
    gated[full == 0] = 0
    return gated


def paste_crop_mask(mask: np.ndarray, full_hw: tuple[int, int], roi: Roi) -> np.ndarray:
    """Place a crop-sized mask back on the full frame."""
    full = np.zeros(full_hw, np.uint8)
    crop = mask
    if crop.ndim == 3:
        crop = crop[:, :, 0]
    if crop.shape[:2] != (roi.height, roi.width):
        crop = cv2.resize(crop, (roi.width, roi.height), interpolation=cv2.INTER_NEAREST)
    full[roi.y0 : roi.y1, roi.x0 : roi.x1] = np.where(crop > 0, 255, 0).astype(np.uint8)
    return full


def resolve_instances(
    candidates: list[tuple[str, float, np.ndarray]],
    shape: tuple[int, int],
    *,
    dedup_iou: float,
    min_area: int,
) -> tuple[np.ndarray, list[FinalInstance]]:
    """Return the label map and the instances that survived dedup and overlap."""
    ordered = sorted(candidates, key=lambda item: (-item[1], item[0]))
    kept: list[tuple[str, float, np.ndarray]] = []
    for class_name, score, mask in ordered:
        if mask.shape != shape:
            raise ValueError(f"Mask shape {mask.shape} does not match the image {shape}.")
        binary = mask > 0
        if int(binary.sum()) < min_area:
            continue
        if any(_mask_iou(binary, previous) >= dedup_iou for _, _, previous in kept):
            continue
        kept.append((class_name, score, binary))
    label = np.zeros(shape, np.uint16)
    painted: list[tuple[str, float]] = []
    for class_name, score, binary in kept:
        free = binary & (label == 0)
        if int(free.sum()) < min_area:
            continue
        label[free] = len(painted) + 1
        painted.append((class_name, score))
    final_label = np.zeros(shape, np.uint16)
    instances: list[FinalInstance] = []
    next_id = 1
    for index, (class_name, score) in enumerate(painted, start=1):
        pixels = label == index
        polygon = _contour(pixels)
        if polygon is None or int(pixels.sum()) < min_area:
            continue
        final_label[pixels] = next_id
        instances.append(FinalInstance(next_id, class_name, score, polygon))
        next_id += 1
    if next_id - 1 <= 255:
        return final_label.astype(np.uint8), instances
    return final_label, instances


def counts_for(instances: list[FinalInstance], class_names: tuple[str, ...]) -> dict[str, int]:
    counts = {name: 0 for name in class_names}
    for instance in instances:
        if instance.class_name not in counts:
            raise KeyError(instance.class_name)
        counts[instance.class_name] += 1
    return counts


def defect_ratio(counts: dict[str, int], good_name: str) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return (total - counts.get(good_name, 0)) / total


def _contour(pixels: np.ndarray) -> list[list[int]] | None:
    binary = pixels.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    return [[int(x), int(y)] for x, y in points]


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    if intersection == 0:
        return 0.0
    union = int(np.count_nonzero(left)) + int(np.count_nonzero(right)) - intersection
    if union <= 0:
        return 0.0
    return intersection / union
