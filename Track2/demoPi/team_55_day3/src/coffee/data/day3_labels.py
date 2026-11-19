"""Turn string polygon labels into a YOLO-seg tree, plus separated-set crops.

The mixed labels.json stores submission class names, not integer ids.
Train ids come from the class registry. Splits are by image, never by bean.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from coffee.config import AppConfig
from coffee.data.beans import Instance
from coffee.data.extract import Extractor
from coffee.data.labels import read_yolo_seg, write_yolo_seg
from coffee.data.validate import DatasetError

logger = logging.getLogger("coffee.data.day3")

_LEAKAGE = (
    "Split is by image. The same beans were photographed from different angles, "
    "so a val score leaks identity and is optimistic."
)
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (40, 80, 180),
    (0, 200, 255),
    (60, 180, 60),
    (220, 180, 40),
    (180, 80, 200),
    (30, 30, 30),
    (40, 40, 220),
)


@dataclass(frozen=True)
class BeanLabel:
    class_name: str
    class_id: int
    polygon_xy: np.ndarray
    source_id: int


@dataclass(frozen=True)
class ImageLabels:
    image_name: str
    beans: tuple[BeanLabel, ...]


def run_day3(cfg: AppConfig) -> Path:
    """Write the YOLO tree, the fold file, previews, and separated crops."""
    destination = prepare_labeled_set(cfg)
    counts = harvest_separated(cfg, Extractor(cfg).extract)
    logger.info("Separated crops: %s", ", ".join(f"{name}={counts[name]}" for name in cfg.classes))
    logger.warning("%s", _LEAKAGE)
    return destination


def prepare_labeled_set(cfg: AppConfig) -> Path:
    """Convert the mixed string labels. Does not run the bean extractor."""
    label_path = cfg.data_dir / cfg.mixed_dirname / cfg.labels_filename
    images = load_string_labels(label_path, cfg)
    names = [item.image_name for item in images]
    train_names, val_names = image_holdout(names, cfg.day3_val_fraction, cfg.seed)
    folds = image_folds(names, min(cfg.day3_fold_count, len(names)), cfg.seed)
    destination = cfg.output_dir / "day3"
    _reset(destination)
    by_name = {item.image_name: item for item in images}
    train_beans = _write_split(cfg, destination, by_name, train_names, "train")
    val_beans = _write_split(cfg, destination, by_name, val_names, "val")
    _write_data_yaml(destination / "yolo", cfg, has_val=bool(val_names))
    _write_sheet(destination / "preview")
    stats = _stats(cfg, images, train_names, val_names, train_beans, val_beans)
    (destination / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (destination / "folds.json").write_text(
        json.dumps(_fold_payload(cfg, train_names, val_names, folds), indent=2) + "\n",
        encoding="utf-8",
    )
    logger.warning("%s", _LEAKAGE)
    logger.info(
        "Day 3 labels: %s images, %s beans, train %s, val %s, dir %s",
        len(images),
        sum(stats["per_class"].values()),
        len(train_names),
        len(val_names),
        destination,
    )
    logger.info("Per-class beans: %s", ", ".join(f"{name}={stats['per_class'][name]}" for name in cfg.submission_names))
    return destination


def load_string_labels(path: Path, cfg: AppConfig) -> list[ImageLabels]:
    """Read labels.json. Class values are names. Integer ids are rejected."""
    if not path.is_file():
        raise DatasetError(f"Label file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise DatasetError(f"{path} must be a non-empty list of images.")
    images: list[ImageLabels] = []
    seen: set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            raise DatasetError(f"{path} has an entry that is not an object.")
        image_name = str(entry.get("image_name", "")).strip()
        if image_name == "" or image_name in seen:
            raise DatasetError(f"Missing or duplicate image_name in {path}: {image_name!r}.")
        seen.add(image_name)
        objects = entry.get("objects")
        if not isinstance(objects, list) or not objects:
            raise DatasetError(f"{image_name} has no objects.")
        beans: list[BeanLabel] = []
        for obj in objects:
            beans.append(_bean(image_name, obj, cfg))
        images.append(ImageLabels(image_name, tuple(beans)))
    return images


def image_holdout(names: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Shuffle images, then cut a val slice. No image is in both lists."""
    ordered = _shuffled(names, seed)
    if len(ordered) < 2 or val_fraction <= 0:
        return sorted(ordered), []
    val_count = min(len(ordered) - 1, max(1, int(round(len(ordered) * val_fraction))))
    return sorted(ordered[val_count:]), sorted(ordered[:val_count])


def image_folds(names: list[str], fold_count: int, seed: int) -> list[tuple[list[str], list[str]]]:
    """Image-level k-fold. Each item is (train names, val names)."""
    if fold_count < 2:
        raise DatasetError(f"fold_count must be >= 2, got {fold_count}.")
    ordered = _shuffled(names, seed)
    count = min(fold_count, len(ordered))
    if count < 2:
        raise DatasetError("Need at least 2 images for a k-fold split.")
    folds = [ordered[index::count] for index in range(count)]
    assigned: list[tuple[list[str], list[str]]] = []
    for val in folds:
        val_set = set(val)
        train = [name for name in ordered if name not in val_set]
        assigned.append((sorted(train), sorted(val)))
    return assigned


def harvest_separated(
    cfg: AppConfig,
    extract: Callable[[np.ndarray, str], list[Instance]],
) -> dict[str, int]:
    """Cut bean crops from the separated class folders. Skip the mixed copy and duplicate bytes."""
    counts = {name: 0 for name in cfg.classes}
    seen_hash: set[str] = set()
    crops_root = cfg.output_dir / "day3" / "crops" / "separated"
    found_any = False
    for dirname in cfg.separated_dirnames:
        root = cfg.data_dir / dirname
        if not root.is_dir():
            raise DatasetError(f"Separated dataset folder not found: {root}")
        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if class_dir.name == cfg.mixed_dirname:
                logger.info("Skipping %s. Those frames are the labeled mix.", class_dir)
                continue
            class_name = resolve_class_name(class_dir.name, cfg)
            images = _images_in(class_dir, cfg)
            if not images:
                raise DatasetError(f"No images in separated class folder {class_dir}.")
            found_any = True
            for image_path in images:
                digest = _file_sha256(image_path)
                if digest in seen_hash:
                    logger.info("Skipping duplicate bytes %s", image_path)
                    continue
                seen_hash.add(digest)
                image = _read_image(image_path)
                found = extract(image, class_name)
                logger.info("Harvested %s beans from %s (%s)", len(found), image_path.name, class_name)
                for instance in found:
                    if instance.class_name != class_name:
                        raise DatasetError(
                            f"Extractor returned class {instance.class_name!r} for folder {class_name!r}."
                        )
                    _write_crop(instance, crops_root / class_name / f"{counts[class_name]:04d}.png")
                    counts[class_name] += 1
    if not found_any:
        raise DatasetError("No separated class folders were found.")
    empty = [name for name in cfg.classes if counts[name] == 0]
    if empty:
        raise DatasetError("Separated harvest found no beans for: " + ", ".join(empty) + ".")
    return counts


def resolve_class_name(name: str, cfg: AppConfig) -> str:
    text = name.strip()
    if text in cfg.class_to_id:
        return text
    mapped = cfg.class_aliases.get(text)
    if mapped is not None:
        return mapped
    key = "".join(char for char in text.lower() if char.isalnum())
    alias = cfg.sample_aliases.get(key)
    if alias is not None and alias in cfg.class_to_id:
        return alias
    known = ", ".join(cfg.classes)
    raise DatasetError(f"Unknown class {name!r}. Expected one of: {known}.")


def _bean(image_name: str, obj: object, cfg: AppConfig) -> BeanLabel:
    if not isinstance(obj, dict):
        raise DatasetError(f"{image_name} has an object that is not a mapping.")
    raw_class = obj.get("class")
    if not isinstance(raw_class, str):
        raise DatasetError(
            f"{image_name} class must be a string name, got {type(raw_class).__name__}. "
            "This file does not store integer class ids."
        )
    class_name = resolve_class_name(raw_class, cfg)
    polygon = _polygon(image_name, obj.get("polygon"))
    source_id = obj.get("id")
    if not isinstance(source_id, int):
        raise DatasetError(f"{image_name} object id must be an integer.")
    return BeanLabel(class_name, cfg.class_to_id[class_name], polygon, source_id)


def _polygon(image_name: str, raw: object) -> np.ndarray:
    if not isinstance(raw, list) or len(raw) < 3:
        raise DatasetError(f"{image_name} polygon needs at least 3 points.")
    points: list[list[float]] = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise DatasetError(f"{image_name} polygon point must be [x, y].")
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"{image_name} polygon point is not numeric.") from exc
    return np.asarray(points, dtype=np.float32)


def _write_split(
    cfg: AppConfig,
    destination: Path,
    by_name: dict[str, ImageLabels],
    names: list[str],
    split: str,
) -> int:
    total = 0
    for image_name in names:
        labels = by_name[image_name]
        source = cfg.data_dir / cfg.mixed_dirname / image_name
        image = _read_image(source)
        height, width = image.shape[:2]
        class_ids: list[int] = []
        polygons: list[np.ndarray] = []
        for bean in labels.beans:
            polygon = _clip_polygon(image_name, bean.polygon_xy, width, height)
            class_ids.append(bean.class_id)
            polygons.append(polygon)
            _write_crop(
                _polygon_crop(image, polygon, bean.class_name),
                destination / "crops" / "labeled" / bean.class_name / f"{Path(image_name).stem}_{bean.source_id:02d}.png",
            )
        stem = Path(image_name).stem
        image_dest = destination / "yolo" / "images" / split / image_name
        label_dest = destination / "yolo" / "labels" / split / f"{stem}.txt"
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_dest)
        write_yolo_seg(label_dest, class_ids, polygons, width, height)
        written_ids, written_polygons = read_yolo_seg(label_dest, width, height)
        if written_ids != class_ids or len(written_polygons) != len(polygons):
            raise DatasetError(f"YOLO label round-trip failed for {image_name}.")
        _write_preview(cfg, destination / "preview" / f"{stem}.jpg", image, labels)
        total += len(labels.beans)
        logger.info("%s %s beans=%s size=%sx%s", split, image_name, len(labels.beans), width, height)
    return total


def _clip_polygon(image_name: str, polygon: np.ndarray, width: int, height: int) -> np.ndarray:
    if polygon.shape[0] < 3:
        raise DatasetError(f"{image_name} polygon has fewer than 3 points.")
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    if float(xs.min()) < -1 or float(ys.min()) < -1 or float(xs.max()) > width or float(ys.max()) > height:
        raise DatasetError(f"{image_name} polygon is outside the image ({width}x{height}).")
    clipped = polygon.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    return clipped


def _polygon_crop(image: np.ndarray, polygon: np.ndarray, class_name: str) -> Instance:
    mask = np.zeros(image.shape[:2], np.uint8)
    contour = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [contour], 255)
    x, y, box_w, box_h = cv2.boundingRect(contour)
    pad = 2
    height, width = mask.shape
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + box_w + pad)
    y1 = min(height, y + box_h + pad)
    return Instance(class_name, polygon, image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy())


def _write_preview(cfg: AppConfig, path: Path, image: np.ndarray, labels: ImageLabels) -> None:
    height, width = image.shape[:2]
    scale = min(1.0, cfg.day3_preview_max_side / float(max(height, width)))
    canvas = image
    if scale < 1:
        canvas = cv2.resize(image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))))
    overlay = canvas.copy()
    for bean in labels.beans:
        points = np.round(bean.polygon_xy * scale).astype(np.int32).reshape(-1, 1, 2)
        color = _PALETTE[bean.class_id % len(_PALETTE)]
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(canvas, [points], True, color, 1, cv2.LINE_AA)
        x, y = points[0, 0]
        cv2.putText(canvas, bean.class_name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    blended = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), blended):
        raise DatasetError(f"Could not write preview: {path}")


def _write_sheet(preview: Path) -> None:
    paths = sorted(path for path in preview.glob("*.jpg") if path.name != "sheet.jpg")
    if not paths:
        return
    thumb_h = 240
    thumbs: list[np.ndarray] = []
    for path in paths:
        image = _read_image(path)
        scale = thumb_h / float(image.shape[0])
        thumbs.append(cv2.resize(image, (max(1, int(round(image.shape[1] * scale))), thumb_h)))
    columns = 4
    rows = math.ceil(len(thumbs) / columns)
    cell_w = max(int(thumb.shape[1]) for thumb in thumbs)
    sheet = np.zeros((rows * thumb_h, columns * cell_w, 3), np.uint8)
    for index, thumb in enumerate(thumbs):
        row, column = divmod(index, columns)
        y = row * thumb_h
        x = column * cell_w
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    destination = preview / "sheet.jpg"
    if not cv2.imwrite(str(destination), sheet):
        raise DatasetError(f"Could not write preview sheet: {destination}")


def _stats(
    cfg: AppConfig,
    images: list[ImageLabels],
    train_names: list[str],
    val_names: list[str],
    train_beans: int,
    val_beans: int,
) -> dict[str, object]:
    per_class = {name: 0 for name in cfg.submission_names}
    per_image: list[dict[str, object]] = []
    for item in images:
        counts = {name: 0 for name in cfg.submission_names}
        for bean in item.beans:
            counts[bean.class_name] += 1
            per_class[bean.class_name] += 1
        per_image.append({"image_name": item.image_name, "beans": len(item.beans), "counts": counts})
    return {
        "images": len(images),
        "beans": sum(per_class.values()),
        "per_class": per_class,
        "per_image": per_image,
        "holdout": {"train": train_names, "val": val_names, "train_beans": train_beans, "val_beans": val_beans},
        "leakage_warning": _LEAKAGE,
    }


def _fold_payload(
    cfg: AppConfig,
    train_names: list[str],
    val_names: list[str],
    folds: list[tuple[list[str], list[str]]],
) -> dict[str, object]:
    return {
        "seed": cfg.seed,
        "leakage_warning": _LEAKAGE,
        "holdout": {"train": train_names, "val": val_names},
        "folds": [{"train": train, "val": val} for train, val in folds],
    }


def _write_data_yaml(yolo_root: Path, cfg: AppConfig, *, has_val: bool) -> None:
    payload = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/val" if has_val else "images/train",
        "names": {index: name for index, name in enumerate(cfg.classes)},
    }
    yolo_root.mkdir(parents=True, exist_ok=True)
    (yolo_root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _shuffled(names: list[str], seed: int) -> list[str]:
    if len(set(names)) != len(names):
        raise DatasetError("Image list contains a duplicate name.")
    ordered = sorted(names)
    np.random.default_rng(seed).shuffle(ordered)
    return ordered


def _images_in(folder: Path, cfg: AppConfig) -> list[Path]:
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in cfg.image_extensions
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetError(f"Could not read image: {path}")
    return image


def _write_crop(instance: Instance, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgra = cv2.cvtColor(instance.crop_bgr, cv2.COLOR_BGR2BGRA)
    alpha = instance.crop_mask
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    if alpha.shape[:2] != bgra.shape[:2]:
        raise DatasetError(f"Crop mask size does not match the crop: {path}")
    bgra[:, :, 3] = alpha
    if not cv2.imwrite(str(path), bgra):
        raise DatasetError(f"Could not write crop: {path}")


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
