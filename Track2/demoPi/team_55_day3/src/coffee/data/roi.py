"""Adaptive ROI for gray paper with a vignette.

Beans are locally darker than a heavy blur of the L channel. A border colour
model fails here: the bottom of the page is darker than the centre, so the
border colour marks the bright paper as foreground. Components that touch the
frame are dropped. If nothing remains, the crop falls back to the centre.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Roi:
    x0: int
    y0: int
    x1: int
    y1: int
    source: str

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def estimate_roi(
    image: np.ndarray,
    *,
    dark_delta: float = 12.0,
    margin_frac: float = 0.15,
    fallback_frac: float = 0.7,
) -> Roi:
    """Return an inclusive-exclusive crop box in full-image pixels."""
    height, width = image.shape[:2]
    box = _paper_box(image, dark_delta)
    if box is None:
        return _center(width, height, fallback_frac)
    x0, y0, x1, y1 = box
    pad_x = (x1 - x0) * margin_frac
    pad_y = (y1 - y0) * margin_frac
    x0 = max(0, int(np.floor(x0 - pad_x)))
    y0 = max(0, int(np.floor(y0 - pad_y)))
    x1 = min(width, int(np.ceil(x1 + pad_x)))
    y1 = min(height, int(np.ceil(y1 + pad_y)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return _center(width, height, fallback_frac)
    return Roi(x0, y0, x1, y1, "paper")


def crop_image(image: np.ndarray, roi: Roi) -> np.ndarray:
    return image[roi.y0 : roi.y1, roi.x0 : roi.x1].copy()


def shift_polygon(polygon: np.ndarray, roi: Roi) -> np.ndarray | None:
    """Move a full-image polygon into the crop. Drop it when clipping leaves fewer than 3 points."""
    shifted = polygon.astype(np.float32).copy()
    shifted[:, 0] -= roi.x0
    shifted[:, 1] -= roi.y0
    inside = (
        (shifted[:, 0] >= -1)
        & (shifted[:, 1] >= -1)
        & (shifted[:, 0] <= roi.width)
        & (shifted[:, 1] <= roi.height)
    )
    if int(inside.sum()) < 3:
        return None
    shifted[:, 0] = np.clip(shifted[:, 0], 0, roi.width - 1)
    shifted[:, 1] = np.clip(shifted[:, 1], 0, roi.height - 1)
    unique = np.unique(np.round(shifted).astype(np.int32), axis=0)
    if unique.shape[0] < 3:
        return None
    return shifted


def _paper_box(image: np.ndarray, dark_delta: float) -> tuple[float, float, float, float] | None:
    height, width = image.shape[:2]
    long_side = max(height, width)
    scale = min(1.0, 1024.0 / float(long_side))
    work_w = max(1, int(round(width * scale)))
    work_h = max(1, int(round(height * scale)))
    small = image if scale == 1 else cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
    lightness = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    sigma = max(3.0, 0.034 * max(work_w, work_h))
    paper = cv2.GaussianBlur(lightness, (0, 0), sigma)
    mask = ((paper - lightness) > dark_delta).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    band = max(2, int(round(0.015 * min(work_w, work_h))))
    max_area = 0.15 * work_w * work_h
    keep = np.zeros(mask.shape, np.uint8)
    for index in range(1, count):
        x, y, box_w, box_h, area = (int(v) for v in stats[index])
        if area < 25 or area > max_area:
            continue
        if x <= band or y <= band or x + box_w >= work_w - band or y + box_h >= work_h - band:
            continue
        keep[labels == index] = 1
    ys, xs = np.where(keep > 0)
    if xs.size < 50:
        return None
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    return float(x0 / scale), float(y0 / scale), float(x1 / scale), float(y1 / scale)


def _center(width: int, height: int, frac: float) -> Roi:
    margin_x = int(round(width * (1.0 - frac) / 2))
    margin_y = int(round(height * (1.0 - frac) / 2))
    return Roi(margin_x, margin_y, width - margin_x, height - margin_y, "center")
