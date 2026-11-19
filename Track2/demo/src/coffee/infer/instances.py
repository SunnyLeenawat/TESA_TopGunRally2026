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


def fit_roi_mask(mask: np.ndarray, roi: Roi) -> np.ndarray:
    """Resize one detection mask to the ROI. Pixels stay 0 or 255."""
    crop = mask[:, :, 0] if mask.ndim == 3 else mask
    if crop.shape[:2] != (roi.height, roi.width):
        crop = cv2.resize(crop, (roi.width, roi.height), interpolation=cv2.INTER_NEAREST)
    return np.where(crop > 0, 255, 0).astype(np.uint8)


def paste_crop_mask(mask: np.ndarray, full_hw: tuple[int, int], roi: Roi) -> np.ndarray:
    """Place a crop-sized mask back on the full frame."""
    full = np.zeros(full_hw, np.uint8)
    full[roi.y0 : roi.y1, roi.x0 : roi.x1] = fit_roi_mask(mask, roi)
    return full


def resolve_on_roi(
    candidates: list[tuple[str, float, np.ndarray]],
    roi: Roi,
    full_hw: tuple[int, int],
    *,
    dedup_iou: float,
    min_area: int,
) -> tuple[np.ndarray, list[FinalInstance]]:
    """Resolve inside the ROI, then return a full-frame label and full-image polygons.

    A one-pixel zero pad is added on every side that does not touch the photo
    edge. Contours of a mask that ends on the crop border then match the
    full-frame contour, and the pad is removed before the points are shifted.
    """
    height, width = full_hw
    if roi.height <= 0 or roi.width <= 0:
        raise ValueError(f"ROI has no area: {roi}.")
    pad_top = 1 if roi.y0 > 0 else 0
    pad_left = 1 if roi.x0 > 0 else 0
    pad_bottom = 1 if roi.y1 < height else 0
    pad_right = 1 if roi.x1 < width else 0
    padded: list[tuple[str, float, np.ndarray]] = []
    for class_name, score, mask in candidates:
        if mask.shape != (roi.height, roi.width):
            raise ValueError(f"Mask shape {mask.shape} does not match the ROI {(roi.height, roi.width)}.")
        if pad_top or pad_bottom or pad_left or pad_right:
            mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)))
        padded.append((class_name, score, mask))
    label, instances = resolve_instances(
        padded,
        (roi.height + pad_top + pad_bottom, roi.width + pad_left + pad_right),
        dedup_iou=dedup_iou,
        min_area=min_area,
    )
    crop_label = label[pad_top : pad_top + roi.height, pad_left : pad_left + roi.width]
    full = np.zeros(full_hw, crop_label.dtype)
    full[roi.y0 : roi.y1, roi.x0 : roi.x1] = crop_label
    shifted = [
        FinalInstance(
            item.id,
            item.class_name,
            item.score,
            [[x - pad_left + roi.x0, y - pad_top + roi.y0] for x, y in item.polygon],
        )
        for item in instances
    ]
    return full, shifted


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
