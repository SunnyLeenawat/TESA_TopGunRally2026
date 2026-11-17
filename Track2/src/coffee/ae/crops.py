"""Turn a masked bean into the square the autoencoder was trained on."""

from __future__ import annotations

import cv2
import numpy as np

_PAD = 114


def composite(bgra: np.ndarray, size: int) -> np.ndarray:
    """Scale a BGRA crop so the longer side fits ``size``, centered on gray 114."""
    if bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError(f"Expected a BGRA crop, got shape {bgra.shape}.")
    color = bgra[:, :, :3].astype(np.float32)
    alpha = bgra[:, :, 3].astype(np.float32) / 255.0
    height, width = color.shape[:2]
    scale = size / max(height, width, 1)
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    color = cv2.resize(color, (resized_w, resized_h), interpolation=interpolation)
    alpha = cv2.resize(alpha, (resized_w, resized_h), interpolation=interpolation)[:, :, None]
    canvas = np.full((size, size, 3), _PAD, np.float32)
    top = (size - resized_h) // 2
    left = (size - resized_w) // 2
    region = canvas[top : top + resized_h, left : left + resized_w]
    canvas[top : top + resized_h, left : left + resized_w] = color * alpha + region * (1.0 - alpha)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def masked_square(image_bgr: np.ndarray, mask: np.ndarray, box_xyxy: np.ndarray, size: int) -> np.ndarray:
    """Cut one detection out of a BGR image and composite it for the encoder."""
    height, width = image_bgr.shape[:2]
    x1 = max(0, int(np.floor(float(box_xyxy[0]))))
    y1 = max(0, int(np.floor(float(box_xyxy[1]))))
    x2 = min(width, int(np.ceil(float(box_xyxy[2]))))
    y2 = min(height, int(np.ceil(float(box_xyxy[3]))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detection box is empty.")
    crop = image_bgr[y1:y2, x1:x2]
    alpha = mask[y1:y2, x1:x2]
    if alpha.ndim != 2 or alpha.shape[:2] != crop.shape[:2]:
        raise ValueError("Mask crop does not match the detection box.")
    if int(alpha.max()) <= 1:
        alpha = alpha * 255
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha.astype(np.uint8)
    return composite(bgra, size)
