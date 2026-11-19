"""Pull individual beans out of a photo.

FastSAM is the default. On these sample photos the beans and the wooden table
share a hue, so a saturation threshold keeps the wood and drops the beans.
Classical Lab distance remains for high-contrast images and for unit tests.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

from coffee.config import AppConfig, ConfigError
from coffee.data.beans import Instance, mask_to_instance, nms

logger = logging.getLogger("coffee.data.extract")


class Extractor:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._model: object | None = None

    def extract(self, image: np.ndarray, class_name: str) -> list[Instance]:
        if self.cfg.extract_backend == "classical":
            masks = _classical_masks(image, self.cfg.extract_lab_distance)
        else:
            masks = self._fastsam_masks(image)
        instances: list[Instance] = []
        for mask in masks:
            instance = mask_to_instance(
                image,
                mask,
                class_name,
                min_area_frac=self.cfg.extract_min_area_frac,
                max_area_frac=self.cfg.extract_max_area_frac,
                max_aspect=self.cfg.extract_max_aspect,
            )
            if instance is not None:
                instances.append(instance)
        return nms(instances)

    def _fastsam_masks(self, image: np.ndarray) -> list[np.ndarray]:
        model = self._load_fastsam()
        device: int | str = self.cfg.device
        if str(device).isdigit():
            device = int(device)
        results = model.predict(  # type: ignore[attr-defined]
            image,
            device=device,
            imgsz=self.cfg.extract_imgsz,
            conf=self.cfg.extract_conf,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]
        if result.masks is None:
            return []
        height, width = image.shape[:2]
        masks: list[np.ndarray] = []
        data = result.masks.data.cpu().numpy()
        for mask in data:
            binary = (mask > 0.5).astype(np.uint8) * 255
            if binary.shape != (height, width):
                binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
            masks.append(binary)
        return masks

    def _load_fastsam(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import FastSAM
        except ImportError as exc:
            raise ConfigError(
                "extract.backend is fastsam, but ultralytics is not installed. "
                "Run this step in the training container."
            ) from exc
        weights = _ensure_weights(self.cfg)
        logger.info("Loading FastSAM weights %s", weights)
        self._model = FastSAM(str(weights))
        return self._model


def _ensure_weights(cfg: AppConfig) -> Path:
    destination = cfg.model_dir / cfg.extract_weights
    if destination.is_file():
        return destination
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import FastSAM

    logger.info("Downloading %s into %s", cfg.extract_weights, destination)
    FastSAM(cfg.extract_weights)
    downloaded = Path(cfg.extract_weights)
    if not downloaded.is_file():
        raise ConfigError(f"FastSAM download did not produce {cfg.extract_weights}.")
    shutil.copy2(downloaded, destination)
    return destination


def _classical_masks(image: np.ndarray, lab_distance: float) -> list[np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    border = max(4, int(round(0.02 * min(height, width))))
    strip = np.concatenate(
        [
            lab[:border].reshape(-1, 3),
            lab[-border:].reshape(-1, 3),
            lab[:, :border].reshape(-1, 3),
            lab[:, -border:].reshape(-1, 3),
        ],
        axis=0,
    )
    wood = np.median(strip, axis=0)
    distance = np.linalg.norm(lab - wood, axis=2)
    mask = (distance > lab_distance).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return _watershed_masks(image, mask)


def _watershed_masks(image: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    if int(mask.sum()) == 0:
        return []
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peak = float(distance.max())
    if peak <= 0:
        return []
    sure = (distance > 0.35 * peak).astype(np.uint8)
    count, markers = cv2.connectedComponents(sure)
    markers = markers + 1
    markers[mask == 0] = 0
    flooded = cv2.watershed(image.copy(), markers.astype(np.int32))
    masks: list[np.ndarray] = []
    for label in range(2, count + 1):
        component = np.where(flooded == label, 255, 0).astype(np.uint8)
        if int(component.sum()) > 0:
            masks.append(component)
    return masks
