"""Copy real bean crops onto gray paper taken from the training photos.

Later beans cover earlier ones. The label polygon is only the visible part.
Rare classes are drawn more often. Real photos stay in the training list.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from coffee.config import AppConfig, ConfigError
from coffee.data.day3_labels import load_string_labels
from coffee.data.labels import write_yolo_seg

logger = logging.getLogger("coffee.data.piles")


@dataclass(frozen=True)
class Crop:
    class_id: int
    bgr: np.ndarray
    mask: np.ndarray


def class_weights(counts: dict[str, int]) -> dict[str, float]:
    """Inverse-frequency weights. A missing class gets no draws."""
    raw = {name: (1.0 / count if count > 0 else 0.0) for name, count in counts.items()}
    total = sum(raw.values())
    if total <= 0:
        raise ConfigError("No crops are available for the pile generator.")
    return {name: value / total for name, value in raw.items()}


def paste_sprite(
    image: np.ndarray,
    owner: np.ndarray,
    sprite_bgr: np.ndarray,
    sprite_mask: np.ndarray,
    origin_xy: tuple[int, int],
    bean_index: int,
) -> int:
    """Paint one sprite. Return how many of its pixels stay visible. Rolls back a tiny paste."""
    x0, y0 = origin_xy
    height, width = owner.shape
    sprite_h, sprite_w = sprite_mask.shape[:2]
    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(width, x0 + sprite_w)
    dst_y1 = min(height, y0 + sprite_h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return 0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    local_mask = sprite_mask[src_y0:src_y1, src_x0:src_x1] > 0
    if int(local_mask.sum()) < 40:
        return 0
    previous_owner = owner[dst_y0:dst_y1, dst_x0:dst_x1].copy()
    previous_image = image[dst_y0:dst_y1, dst_x0:dst_x1].copy()
    shadow = np.zeros(local_mask.shape, np.uint8)
    cv2.ellipse(
        shadow,
        (max(1, shadow.shape[1] // 2 + 3), max(1, shadow.shape[0] // 2 + 5)),
        (max(1, shadow.shape[1] // 3), max(1, shadow.shape[0] // 4)),
        0,
        0,
        360,
        255,
        -1,
    )
    shadow_pixels = (shadow > 0) & (previous_owner == 0) & ~local_mask
    region = image[dst_y0:dst_y1, dst_x0:dst_x1]
    region[shadow_pixels] = (region[shadow_pixels].astype(np.float32) * 0.75).astype(np.uint8)
    region[local_mask] = sprite_bgr[src_y0:src_y1, src_x0:src_x1][local_mask]
    owner_region = owner[dst_y0:dst_y1, dst_x0:dst_x1]
    owner_region[local_mask] = bean_index
    visible = int(np.count_nonzero(owner[dst_y0:dst_y1, dst_x0:dst_x1] == bean_index))
    if visible < 40 or visible < 0.25 * int(local_mask.sum()):
        owner[dst_y0:dst_y1, dst_x0:dst_x1] = previous_owner
        image[dst_y0:dst_y1, dst_x0:dst_x1] = previous_image
        return 0
    return visible


def build_piles(cfg: AppConfig) -> Path:
    """Write synthetic piles and a train bundle that still contains every real training image."""
    crops = _load_crops(cfg)
    backgrounds = _backgrounds(cfg)
    rng = np.random.default_rng(cfg.seed + 7)
    weights = class_weights({name: len(crops[name]) for name in cfg.classes})
    logger.info("Pile class weights: %s", ", ".join(f"{name}={weights[name]:.3f}" for name in cfg.classes))
    pile_root = cfg.output_dir / "day3" / "piles"
    if pile_root.exists():
        shutil.rmtree(pile_root)
    (pile_root / "images").mkdir(parents=True)
    (pile_root / "labels").mkdir(parents=True)
    totals = {name: 0 for name in cfg.classes}
    names = list(cfg.classes)
    probabilities = np.array([weights[name] for name in names], dtype=np.float64)
    for index in range(cfg.day3_pile_count):
        canvas = backgrounds[int(rng.integers(0, len(backgrounds)))].copy()
        wanted = int(rng.integers(cfg.day3_pile_beans_min, cfg.day3_pile_beans_max + 1))
        class_ids, polygons = _fill_pile(canvas, crops, names, probabilities, wanted, rng)
        stem = f"pile_{index:03d}"
        if not cv2.imwrite(str(pile_root / "images" / f"{stem}.jpg"), canvas):
            raise ConfigError(f"Could not write pile {stem}.")
        write_yolo_seg(pile_root / "labels" / f"{stem}.txt", class_ids, polygons, canvas.shape[1], canvas.shape[0])
        for class_id in class_ids:
            totals[cfg.classes[class_id]] += 1
        logger.info("Pile %s beans=%s", stem, len(class_ids))
    (pile_root / "stats.json").write_text(
        json.dumps({"images": cfg.day3_pile_count, "per_class": totals, "weights": weights}, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Pile beans: %s", ", ".join(f"{name}={totals[name]}" for name in cfg.classes))
    _mix_bundle(cfg, pile_root)
    return pile_root


def _fill_pile(canvas, crops, names, probabilities, wanted, rng) -> tuple[list[int], list[np.ndarray]]:
    owner = np.zeros(canvas.shape[:2], np.int16)
    chosen: list[tuple[int, int]] = []
    attempts = 0
    while len(chosen) < wanted and attempts < wanted * 4:
        attempts += 1
        name = str(rng.choice(names, p=probabilities))
        if not crops[name]:
            continue
        crop = crops[name][int(rng.integers(0, len(crops[name])))]
        brightness = float(rng.uniform(0.85, 1.15))
        sprite_bgr, sprite_mask = _warp(crop, float(rng.uniform(0, 360)), float(rng.uniform(0.85, 1.15)), brightness)
        x0 = int(rng.integers(0, max(1, canvas.shape[1] - 1))) - sprite_mask.shape[1] // 2
        y0 = int(rng.integers(0, max(1, canvas.shape[0] - 1))) - sprite_mask.shape[0] // 2
        bean_index = len(chosen) + 1
        visible = paste_sprite(canvas, owner, sprite_bgr, sprite_mask, (x0, y0), bean_index)
        if visible == 0:
            continue
        chosen.append((bean_index, crop.class_id))
    if int(rng.integers(0, 2)) == 0:
        blurred = cv2.GaussianBlur(canvas, (3, 3), 0)
        canvas[:] = blurred
    class_ids: list[int] = []
    polygons: list[np.ndarray] = []
    for bean_index, class_id in chosen:
        polygon = _owner_polygon(owner, bean_index)
        if polygon is None:
            continue
        class_ids.append(class_id)
        polygons.append(polygon)
    return class_ids, polygons


def _warp(crop: Crop, angle: float, scale: float, brightness: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = crop.mask.shape
    center = ((width - 1) / 2, (height - 1) / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    cosine = abs(float(matrix[0, 0]))
    sine = abs(float(matrix[0, 1]))
    new_w = max(1, int(round(height * sine + width * cosine)))
    new_h = max(1, int(round(height * cosine + width * sine)))
    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]
    bgr = cv2.warpAffine(crop.bgr, matrix, (new_w, new_h))
    mask = cv2.warpAffine(crop.mask, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST)
    bgr = np.clip(bgr.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return bgr, mask


def _owner_polygon(owner: np.ndarray, bean_index: int) -> np.ndarray | None:
    binary = (owner == bean_index).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    return points.astype(np.float32)


def _load_crops(cfg: AppConfig) -> dict[str, list[Crop]]:
    crops = {name: [] for name in cfg.classes}
    roots = [cfg.output_dir / "day3" / "crops" / "labeled", cfg.output_dir / "day3" / "crops" / "separated"]
    for root in roots:
        if not root.is_dir():
            raise ConfigError(f"Missing crops: {root}. Run day3-data before building piles.")
        for name in cfg.classes:
            folder = root / name
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.png")):
                bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
                    logger.warning("Skipping crop that is not BGRA: %s", path)
                    continue
                mask = bgra[:, :, 3]
                if int(np.count_nonzero(mask)) < 80:
                    continue
                height, width = mask.shape
                if min(height, width) < 12 or max(height, width) / float(min(height, width)) > 4:
                    continue
                crops[name].append(Crop(cfg.class_to_id[name], bgra[:, :, :3], mask))
    empty = [name for name, items in crops.items() if not items]
    if empty:
        raise ConfigError("No usable crops for: " + ", ".join(empty))
    logger.info("Loaded crops: %s", ", ".join(f"{name}={len(crops[name])}" for name in cfg.classes))
    return crops


def _backgrounds(cfg: AppConfig) -> list[np.ndarray]:
    labels = load_string_labels(cfg.data_dir / cfg.mixed_dirname / cfg.labels_filename, cfg)
    canvases: list[np.ndarray] = []
    for item in labels:
        image = cv2.imread(str(cfg.data_dir / cfg.mixed_dirname / item.image_name), cv2.IMREAD_COLOR)
        if image is None:
            raise ConfigError(f"Could not read {item.image_name} for a pile background.")
        canvases.append(_paper(image, item, cfg.day3_pile_canvas))
    if not canvases:
        raise ConfigError("No backgrounds could be built.")
    return canvases


def _paper(image: np.ndarray, item, long_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = long_side / float(max(height, width))
    small = cv2.resize(image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))))
    mask = np.zeros(small.shape[:2], np.uint8)
    for bean in item.beans:
        points = np.round(bean.polygon_xy * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [points], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kernel)
    return cv2.inpaint(small, mask, 4, cv2.INPAINT_TELEA)


def _mix_bundle(cfg: AppConfig, pile_root: Path) -> Path:
    """Train on real ROI crops plus piles. Validation stays the real holdout."""
    real = cfg.output_dir / "day3" / "yolo_roi"
    if not (real / "images" / "train").is_dir() or not (real / "images" / "val").is_dir():
        raise ConfigError(f"Missing real ROI bundle {real}. Run day3-train before piles.")
    bundle = cfg.output_dir / "day3" / "yolo_s5"
    if bundle.exists():
        shutil.rmtree(bundle)
    for split in ("train", "val"):
        (bundle / "images" / split).mkdir(parents=True)
        (bundle / "labels" / split).mkdir(parents=True)
        _copy_split(real / "images" / split, real / "labels" / split, bundle, split, "real__")
    _copy_split(pile_root / "images", pile_root / "labels", bundle, "train", "")
    payload = {
        "path": str(bundle.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(cfg.classes)},
    }
    destination = bundle / "data.yaml"
    destination.write_text(__import__("yaml").safe_dump(payload, sort_keys=False), encoding="utf-8")
    train_count = len(list((bundle / "images" / "train").glob("*")))
    val_count = len(list((bundle / "images" / "val").glob("*")))
    logger.info("S5 bundle train=%s val=%s %s", train_count, val_count, destination)
    return destination


def _copy_split(image_dir: Path, label_dir: Path, bundle: Path, split: str, prefix: str) -> None:
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ConfigError(f"Pile image has no label: {image_path}")
        shutil.copy2(image_path, bundle / "images" / split / f"{prefix}{image_path.name}")
        shutil.copy2(label_path, bundle / "labels" / split / f"{prefix}{image_path.stem}.txt")
