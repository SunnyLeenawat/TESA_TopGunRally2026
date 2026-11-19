"""YOLO segmentation polygon text. The only label format this project writes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from coffee.config import ConfigError


def write_yolo_seg(
    path: Path,
    class_ids: list[int],
    polygons_xy: list[np.ndarray],
    width: int,
    height: int,
) -> None:
    if len(class_ids) != len(polygons_xy):
        raise ConfigError("Label class ids and polygons differ in length.")
    lines: list[str] = []
    for class_id, polygon in zip(class_ids, polygons_xy):
        if polygon.shape[0] < 3:
            continue
        coords: list[str] = []
        for x, y in polygon:
            coords.append(f"{float(x) / width:.6f}")
            coords.append(f"{float(y) / height:.6f}")
        lines.append(f"{int(class_id)} " + " ".join(coords))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yolo_seg(path: Path, width: int, height: int) -> tuple[list[int], list[np.ndarray]]:
    if not path.is_file():
        raise ConfigError(f"Label file not found: {path}")
    class_ids: list[int] = []
    polygons: list[np.ndarray] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if text == "":
            continue
        parts = text.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            raise ConfigError(f"{path}:{line_number} is not a YOLO segmentation polygon.")
        try:
            class_id = int(parts[0])
            values = [float(item) for item in parts[1:]]
        except ValueError as exc:
            raise ConfigError(f"{path}:{line_number} has a non-numeric value.") from exc
        points = np.asarray(values, dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        class_ids.append(class_id)
        polygons.append(points)
    return class_ids, polygons


def write_layout(path: Path, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(class_names) + "\n", encoding="utf-8")


def read_layout(path: Path) -> list[str]:
    if not path.is_file():
        raise ConfigError(f"Layout file not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ConfigError(f"Layout file is empty: {path}")
    return names
