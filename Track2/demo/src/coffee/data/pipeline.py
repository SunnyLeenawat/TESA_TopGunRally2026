"""Mock dataset, prep (validate + YOLO split), and extra synthetic piles."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from coffee.config import AppConfig
from coffee.data.beans import Instance
from coffee.data.extract import Extractor
from coffee.data.labels import read_layout, read_yolo_seg, write_yolo_seg
from coffee.data.synth import (
    Placed,
    draw_placed,
    pattern_slots,
    render_assigned,
    render_neat,
    render_pile,
    save_scene,
    tile_wood,
    wood_canvas,
    wood_patch,
)
from coffee.data.validate import DatasetError, DatasetReport, validate_dataset

logger = logging.getLogger("coffee.data")

_NEAT_PATTERNS = ("grid3x3", "circle", "rectangle")
_DAY3_PATTERNS = ("grid3x3", "triangle", "rectangle")


def run_mock(cfg: AppConfig) -> Path:
    samples = _sample_images(cfg)
    extractor = Extractor(cfg)
    crops: dict[str, list[Instance]] = {name: [] for name in cfg.classes}
    photos: list[tuple[np.ndarray, list[Instance]]] = []
    for path in samples:
        class_name = _class_for_sample(path, cfg)
        image = _read_image(path)
        found = extractor.extract(image, class_name)
        logger.info("Extracted %s beans from %s (%s)", len(found), path.name, class_name)
        if not found:
            raise DatasetError(f"No beans extracted from {path}. Check extract.backend and the photo.")
        crops[class_name].extend(found)
        photos.append((image, found))
    missing = [name for name, items in crops.items() if not items]
    if missing:
        raise DatasetError(
            "No bean crops for: " + ", ".join(missing) + ". Add a sample image whose name maps to each class."
        )
    rng = np.random.default_rng(cfg.seed)
    background = _wood_background(photos, cfg.synth_canvas_width, cfg.synth_canvas_height, rng)
    wood_path = cfg.output_dir / "backgrounds" / "wood.jpg"
    wood_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(wood_path), background):
        raise DatasetError(f"Could not write wood background: {wood_path}")
    root = cfg.data_dir / cfg.mock_dirname
    _reset(root)
    _save_crops(crops, cfg.output_dir / "crops")
    preview = cfg.output_dir / "preview" / "mock"
    _reset(preview)
    for class_name, items in crops.items():
        for index in range(cfg.synth_images_per_class):
            pattern = _NEAT_PATTERNS[index % len(_NEAT_PATTERNS)]
            scene, placed = render_neat(background, items, class_name, pattern, rng)
            destination = root / cfg.train_dirname / class_name / f"{pattern}_{index:02d}.jpg"
            _store(scene, placed, destination, cfg, preview, layout=False)
        count = int(rng.integers(cfg.synth_pile_beans_min, cfg.synth_pile_beans_max + 1))
        scene, placed = render_pile(background, crops, [class_name], count, rng)
        destination = root / cfg.test_dirname / class_name / "pile_00.jpg"
        _store(scene, placed, destination, cfg, preview, layout=False)
    for pattern in _DAY3_PATTERNS:
        slots = pattern_slots(pattern, cfg.synth_canvas_width, cfg.synth_canvas_height)
        for index in range(cfg.synth_day3_per_pattern):
            names = _mixed_names(list(cfg.classes), len(slots), cfg.synth_mixed_classes_min, rng)
            scene, placed = render_assigned(background, crops, names, pattern, rng)
            destination = root / cfg.day3_dirname / cfg.train_dirname / f"{pattern}_{index:02d}.jpg"
            _store(scene, placed, destination, cfg, preview, layout=True)
    for index, class_count in enumerate((cfg.synth_mixed_classes_min, len(cfg.classes))):
        names = _mixed_names(list(cfg.classes), class_count, class_count, rng)
        count = int(rng.integers(cfg.synth_pile_beans_min, cfg.synth_pile_beans_max + 1))
        scene, placed = render_pile(background, crops, names, count, rng)
        destination = root / cfg.day3_dirname / cfg.test_dirname / f"pile_{index:02d}.jpg"
        _store(scene, placed, destination, cfg, preview, layout=False)
    report = validate_dataset(root, cfg)
    report.raise_if_problems()
    logger.info("Mock dataset ready at %s", root)
    return root


def run_prep(cfg: AppConfig) -> Path:
    root = _dataset_root(cfg)
    logger.info("Preparing dataset root %s", root)
    report = validate_dataset(root, cfg)
    report.raise_if_problems()
    extractor = Extractor(cfg) if cfg.extract_backend == "classical" else None
    labeled = _ensure_labels(root, cfg, report, extractor)
    yolo_root = cfg.output_dir / "yolo"
    _reset(yolo_root)
    grouped: dict[str, list[Path]] = {}
    for class_name, images in report.train_images.items():
        grouped[class_name] = images
    if report.day3_train:
        grouped["day3"] = report.day3_train
    rng = np.random.default_rng(cfg.seed)
    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for images in grouped.values():
        train_group, val_group = _split_group(images, cfg.synth_val_fraction, rng)
        train_paths.extend(train_group)
        val_paths.extend(val_group)
    for split_name, paths in (("train", train_paths), ("val", val_paths)):
        for image_path in paths:
            _copy_labeled(root, image_path, yolo_root, split_name)
    _write_dataset_yaml(yolo_root, cfg, has_val=bool(val_paths))
    _export_crops_from_labels(labeled, cfg.output_dir / "crops", cfg)
    preview = cfg.output_dir / "preview" / "prep"
    _reset(preview)
    for image_path in train_paths + val_paths:
        _preview_existing_label(image_path, preview, cfg)
    logger.info("YOLO split: %s train images, %s val images, root %s", len(train_paths), len(val_paths), yolo_root)
    if not val_paths:
        logger.warning("Validation split is empty. Add at least 3 images in a class before trusting a val score.")
    return yolo_root


def run_synth(cfg: AppConfig) -> Path:
    crops = _load_crops(cfg.output_dir / "crops", cfg)
    missing = [name for name in cfg.classes if not crops.get(name)]
    if missing:
        raise DatasetError(
            "Missing crops for: " + ", ".join(missing) + ". Run mock-data or prep before synth."
        )
    background = _load_background(cfg, crops)
    rng = np.random.default_rng(cfg.seed + 1)
    destination = cfg.output_dir / "synth"
    _reset(destination)
    preview = destination / "preview"
    for class_name in cfg.classes:
        count = int(rng.integers(cfg.synth_pile_beans_min, cfg.synth_pile_beans_max + 1))
        scene, placed = render_pile(background, crops, [class_name], count, rng)
        _store(scene, placed, destination / "images" / f"{class_name}_pile.jpg", cfg, preview, layout=False)
    for index in range(cfg.synth_day3_per_pattern):
        class_count = cfg.synth_mixed_classes_min + (index % max(1, len(cfg.classes) - cfg.synth_mixed_classes_min + 1))
        class_count = min(len(cfg.classes), max(cfg.synth_mixed_classes_min, class_count))
        names = _mixed_names(list(cfg.classes), class_count, class_count, rng)
        count = int(rng.integers(cfg.synth_pile_beans_min, cfg.synth_pile_beans_max + 1))
        scene, placed = render_pile(background, crops, names, count, rng)
        _store(scene, placed, destination / "images" / f"mixed_{index:02d}.jpg", cfg, preview, layout=False)
    logger.info("Synthetic piles written to %s", destination / "images")
    return destination


def _wood_background(
    photos: list[tuple[np.ndarray, list[Instance]]],
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Use the photo with the most open table. The densest photo is a poor wood source."""
    errors: list[str] = []
    for image, beans in sorted(photos, key=lambda item: len(item[1])):
        try:
            return tile_wood(wood_patch(image, beans), width, height, rng)
        except RuntimeError as exc:
            errors.append(str(exc))
    raise DatasetError("Could not find a clean wood patch in any sample. " + " ".join(errors))


def _sample_images(cfg: AppConfig) -> list[Path]:
    folder = cfg.data_dir / cfg.samples_dirname
    if not folder.is_dir():
        raise DatasetError(f"Missing samples folder: {folder}")
    images = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in cfg.image_extensions
    )
    if not images:
        raise DatasetError(f"No sample images in {folder}. Drop one photo per class there.")
    return images


def _class_for_sample(path: Path, cfg: AppConfig) -> str:
    key = "".join(char for char in path.stem.lower() if char.isalnum())
    if key not in cfg.sample_aliases:
        known = ", ".join(sorted(cfg.sample_aliases))
        raise DatasetError(f"Cannot map sample {path.name} to a class. Known stems: {known}.")
    return cfg.sample_aliases[key]


def _dataset_root(cfg: AppConfig) -> Path:
    """Prefer a real train/ folder. The rehearsal set lives under mock/ until then."""
    if (cfg.data_dir / cfg.train_dirname).is_dir():
        logger.info("Using dataset root %s", cfg.data_dir)
        return cfg.data_dir
    mock_root = cfg.data_dir / cfg.mock_dirname
    if (mock_root / cfg.train_dirname).is_dir():
        logger.info("Using rehearsal root %s", mock_root)
        return mock_root
    return cfg.data_dir


def _ensure_labels(
    root: Path,
    cfg: AppConfig,
    report: DatasetReport,
    extractor: Extractor | None,
) -> list[Path]:
    images: list[Path] = []
    for class_images in report.train_images.values():
        images.extend(class_images)
    images.extend(report.day3_train)
    for image_path in images:
        label_path = image_path.with_suffix(".txt")
        if label_path.is_file():
            continue
        if extractor is None:
            extractor = Extractor(cfg)
        _label_unlabeled(image_path, cfg, extractor)
    return images


def _label_unlabeled(image_path: Path, cfg: AppConfig, extractor: Extractor) -> None:
    image = _read_image(image_path)
    parent = image_path.parent.name
    if parent in cfg.class_to_id:
        found = extractor.extract(image, parent)
        if not found:
            raise DatasetError(f"Extractor found no beans in {image_path}.")
        write_yolo_seg(
            image_path.with_suffix(".txt"),
            [cfg.class_to_id[parent]] * len(found),
            [item.polygon_xy for item in found],
            image.shape[1],
            image.shape[0],
        )
        logger.info("Wrote %s labels for %s", len(found), image_path.name)
        return
    layout_path = image_path.with_name(image_path.stem + cfg.layout_suffix)
    if not layout_path.is_file():
        raise DatasetError(
            f"{image_path} has no YOLO label and no layout file. "
            f"Add {layout_path.name} with one class name per bean, row then column."
        )
    names = read_layout(layout_path)
    unknown = [name for name in names if name not in cfg.class_to_id]
    if unknown:
        raise DatasetError(f"Unknown class names in {layout_path}: {', '.join(unknown)}.")
    found = sorted(extractor.extract(image, names[0]), key=lambda item: (item.center[1], item.center[0]))
    if len(found) != len(names):
        raise DatasetError(
            f"{layout_path} lists {len(names)} beans but the extractor found {len(found)}. "
            "Edit the layout file so the two counts match, in row-major order."
        )
    write_yolo_seg(
        image_path.with_suffix(".txt"),
        [cfg.class_to_id[name] for name in names],
        [item.polygon_xy for item in found],
        image.shape[1],
        image.shape[0],
    )


def _split_group(paths: list[Path], val_fraction: float, rng: np.random.Generator) -> tuple[list[Path], list[Path]]:
    ordered = [paths[int(index)] for index in rng.permutation(len(paths))]
    if len(ordered) < 3 or val_fraction <= 0:
        return ordered, []
    val_count = min(len(ordered) - 1, max(1, int(round(len(ordered) * val_fraction))))
    return ordered[val_count:], ordered[:val_count]


def _copy_labeled(root: Path, image_path: Path, yolo_root: Path, split_name: str) -> None:
    relative = image_path.relative_to(root)
    flat = "__".join(relative.with_suffix("").parts) + image_path.suffix.lower()
    image_dest = yolo_root / "images" / split_name / flat
    label_dest = yolo_root / "labels" / split_name / f"{Path(flat).stem}.txt"
    image_dest.parent.mkdir(parents=True, exist_ok=True)
    label_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_dest)
    shutil.copy2(image_path.with_suffix(".txt"), label_dest)


def _write_dataset_yaml(yolo_root: Path, cfg: AppConfig, *, has_val: bool) -> None:
    payload = {
        "path": str(yolo_root),
        "train": "images/train",
        "val": "images/val" if has_val else "images/train",
        "names": {index: name for index, name in enumerate(cfg.classes)},
    }
    (yolo_root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _export_crops_from_labels(images: list[Path], crops_dir: Path, cfg: AppConfig) -> None:
    _reset(crops_dir)
    counts = {name: 0 for name in cfg.classes}
    for image_path in images:
        image = _read_image(image_path)
        class_ids, polygons = read_yolo_seg(image_path.with_suffix(".txt"), image.shape[1], image.shape[0])
        for class_id, polygon in zip(class_ids, polygons):
            if class_id < 0 or class_id >= len(cfg.classes):
                raise DatasetError(f"Class id {class_id} in {image_path.with_suffix('.txt')} is outside 0-6.")
            class_name = cfg.classes[class_id]
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.round(polygon).astype(np.int32).reshape(-1, 1, 2)], 255)
            x, y, box_w, box_h = cv2.boundingRect(np.round(polygon).astype(np.int32))
            if box_w < 2 or box_h < 2:
                continue
            crop = Instance(
                class_name=class_name,
                polygon_xy=polygon,
                crop_bgr=image[y : y + box_h, x : x + box_w].copy(),
                crop_mask=mask[y : y + box_h, x : x + box_w].copy(),
            )
            _write_crop(crop, crops_dir / class_name / f"{counts[class_name]:04d}.png")
            counts[class_name] += 1
    logger.info("Crops: %s", ", ".join(f"{name}={counts[name]}" for name in cfg.classes))


def _preview_existing_label(image_path: Path, preview: Path, cfg: AppConfig) -> None:
    image = _read_image(image_path)
    class_ids, polygons = read_yolo_seg(image_path.with_suffix(".txt"), image.shape[1], image.shape[0])
    placed = [
        Placed(class_name=cfg.classes[class_id], polygon_xy=polygon)
        for class_id, polygon in zip(class_ids, polygons)
        if 0 <= class_id < len(cfg.classes)
    ]
    destination = preview / f"{image_path.parent.name}__{image_path.stem}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), draw_placed(image, placed))


def _load_crops(folder: Path, cfg: AppConfig) -> dict[str, list[Instance]]:
    crops: dict[str, list[Instance]] = {name: [] for name in cfg.classes}
    if not folder.is_dir():
        return crops
    for class_name in cfg.classes:
        class_dir = folder / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("*.png")):
            bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
                raise DatasetError(f"Crop is not a BGRA PNG: {path}")
            mask = bgra[:, :, 3]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon = contours[0].reshape(-1, 2).astype(np.float32) if contours else np.zeros((0, 2), np.float32)
            crops[class_name].append(
                Instance(
                    class_name=class_name,
                    polygon_xy=polygon,
                    crop_bgr=bgra[:, :, :3],
                    crop_mask=mask,
                )
            )
    return crops


def _load_background(cfg: AppConfig, crops: dict[str, list[Instance]]) -> np.ndarray:
    path = cfg.output_dir / "backgrounds" / "wood.jpg"
    if not path.is_file():
        path = _inpaint_first_labeled(cfg)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetError(f"Could not read wood background: {path}")
    _ = crops
    return wood_canvas(image, cfg.synth_canvas_width, cfg.synth_canvas_height)


def _inpaint_first_labeled(cfg: AppConfig) -> Path:
    root = _dataset_root(cfg)
    labeled = [
        path
        for path in sorted(root.rglob("*.txt"))
        if not path.name.endswith(cfg.layout_suffix)
        and (path.with_suffix(".jpg").is_file() or path.with_suffix(".png").is_file())
    ]
    if not labeled:
        raise DatasetError(
            "No wood background yet. Run mock-data, or add outputs/backgrounds/wood.jpg "
            "(a photo of the empty table)."
        )
    label_path = labeled[0]
    image_path = next(path for path in (label_path.with_suffix(".jpg"), label_path.with_suffix(".png")) if path.is_file())
    image = _read_image(image_path)
    class_ids, polygons = read_yolo_seg(label_path, image.shape[1], image.shape[0])
    instances = [
        Instance(
            class_name=cfg.classes[class_id],
            polygon_xy=polygon,
            crop_bgr=image[:2, :2],
            crop_mask=np.zeros((2, 2), np.uint8),
        )
        for class_id, polygon in zip(class_ids, polygons)
        if 0 <= class_id < len(cfg.classes)
    ]
    background = tile_wood(
        wood_patch(image, instances),
        cfg.synth_canvas_width,
        cfg.synth_canvas_height,
        np.random.default_rng(cfg.seed),
    )
    destination = cfg.output_dir / "backgrounds" / "wood.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), background):
        raise DatasetError(f"Could not write wood background: {destination}")
    return destination


def _save_crops(crops: dict[str, list[Instance]], folder: Path) -> None:
    _reset(folder)
    for class_name, items in crops.items():
        for index, instance in enumerate(items):
            _write_crop(instance, folder / class_name / f"{index:04d}.png")


def _write_crop(instance: Instance, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgra = cv2.cvtColor(instance.crop_bgr, cv2.COLOR_BGR2BGRA)
    alpha = instance.crop_mask
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    bgra[:, :, 3] = alpha
    if not cv2.imwrite(str(path), bgra):
        raise DatasetError(f"Could not write crop: {path}")


def _store(
    scene: np.ndarray,
    placed: list[Placed],
    destination: Path,
    cfg: AppConfig,
    preview: Path,
    *,
    layout: bool,
) -> None:
    if len(placed) < 1:
        raise DatasetError(f"Scene {destination.name} kept no visible beans.")
    save_scene(scene, placed, destination, cfg, layout=layout)
    preview_path = preview / f"{destination.parent.name}__{destination.name}"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), draw_placed(scene, placed))
    logger.info("Wrote %s (%s beans)", destination, len(placed))


def _mixed_names(classes: list[str], count: int, distinct: int, rng: np.random.Generator) -> list[str]:
    distinct = min(len(classes), max(1, distinct), count)
    chosen = [str(item) for item in rng.choice(classes, size=distinct, replace=False)]
    if count > distinct:
        chosen.extend(str(rng.choice(chosen)) for _ in range(count - distinct))
    return [chosen[int(index)] for index in rng.permutation(len(chosen))]


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetError(f"Could not read image: {path}")
    return image


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
