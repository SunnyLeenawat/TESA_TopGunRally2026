"""Fine-tune YOLO26n-seg and export it. NCNN decoding is a separate step."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from coffee.config import AppConfig, ConfigError
from coffee.data.labels import read_yolo_seg, write_yolo_seg
from coffee.data.roi import Roi, crop_image, estimate_roi, shift_polygon

logger = logging.getLogger("coffee.train")


def build_train_bundle(cfg: AppConfig) -> Path:
    """Train on prep images plus synthetic piles. Validation stays the prep split."""
    prepared = cfg.output_dir / "yolo"
    if not (prepared / "data.yaml").is_file():
        raise ConfigError(f"Missing {prepared / 'data.yaml'}. Run prep before training.")
    bundle = cfg.output_dir / "yolo_bundle"
    for split in ("train", "val"):
        image_dir = bundle / "images" / split
        label_dir = bundle / "labels" / split
        if image_dir.exists():
            shutil.rmtree(image_dir)
        if label_dir.exists():
            shutil.rmtree(label_dir)
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        _copy_pair(prepared / "images" / split, prepared / "labels" / split, image_dir, label_dir, "")
    synth_images = cfg.output_dir / "synth" / "images"
    if synth_images.is_dir():
        _copy_pair(
            synth_images,
            synth_images,
            bundle / "images" / "train",
            bundle / "labels" / "train",
            "synth__",
        )
    else:
        logger.warning("No synthetic piles at %s. Training on the prep split only.", synth_images)
    names = yaml.safe_load((prepared / "data.yaml").read_text(encoding="utf-8"))["names"]
    payload = {
        "path": str(bundle),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    destination = bundle / "data.yaml"
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    train_count = len(list((bundle / "images" / "train").glob("*")))
    val_count = len(list((bundle / "images" / "val").glob("*")))
    logger.info("Train bundle: %s train images, %s val images, %s", train_count, val_count, destination)
    return destination


def build_day3_bundle(cfg: AppConfig) -> Path:
    """Crop the Day 3 YOLO tree with the same ROI inference will use."""
    source = cfg.output_dir / "day3" / "yolo"
    data_yaml = source / "data.yaml"
    if not data_yaml.is_file():
        raise ConfigError(f"Missing {data_yaml}. Run day3-data before training.")
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    expected = {index: name for index, name in enumerate(cfg.classes)}
    if {int(key): str(value) for key, value in names.items()} != expected:
        raise ConfigError(f"Day 3 data.yaml names {names} do not match the class registry.")
    bundle = cfg.output_dir / "day3" / "yolo_roi"
    if bundle.exists():
        shutil.rmtree(bundle)
    records: list[dict[str, object]] = []
    dropped = 0
    aspects: list[float] = []
    for split in ("train", "val"):
        image_dir = source / "images" / split
        if not image_dir.is_dir():
            raise ConfigError(f"Missing Day 3 images: {image_dir}")
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            label_path = source / "labels" / split / f"{image_path.stem}.txt"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ConfigError(f"Could not read {image_path}.")
            height, width = image.shape[:2]
            class_ids, polygons = read_yolo_seg(label_path, width, height)
            roi = estimate_roi(
                image,
                dark_delta=cfg.day3_roi_dark_delta,
                margin_frac=cfg.day3_roi_margin_frac,
                fallback_frac=cfg.day3_roi_fallback_frac,
            )
            kept_ids, kept_polygons, lost = _polygons_in_roi(class_ids, polygons, roi)
            dropped += lost
            if not kept_ids:
                raise ConfigError(f"ROI removed every bean from {image_path.name}.")
            cropped = crop_image(image, roi)
            image_dest = bundle / "images" / split / image_path.name
            label_dest = bundle / "labels" / split / f"{image_path.stem}.txt"
            image_dest.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_dest), cropped):
                raise ConfigError(f"Could not write ROI image {image_dest}.")
            write_yolo_seg(label_dest, kept_ids, kept_polygons, roi.width, roi.height)
            aspects.append(roi.height / roi.width)
            records.append(
                {
                    "image": image_path.name,
                    "split": split,
                    "source": roi.source,
                    "x0": roi.x0,
                    "y0": roi.y0,
                    "x1": roi.x1,
                    "y1": roi.y1,
                    "beans": len(kept_ids),
                    "dropped": lost,
                }
            )
            logger.info(
                "ROI %s %s %s %sx%s from %sx%s beans=%s dropped=%s",
                split,
                image_path.name,
                roi.source,
                roi.width,
                roi.height,
                width,
                height,
                len(kept_ids),
                lost,
            )
    payload = {
        "path": str(bundle.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": expected,
    }
    destination = bundle / "data.yaml"
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (bundle / "roi.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "ROI bundle %s. Dropped beans %s. Aspect min %.2f max %.2f. Square imgsz stays %s.",
        destination,
        dropped,
        min(aspects),
        max(aspects),
        cfg.imgsz,
    )
    return destination


def train_day3(cfg: AppConfig) -> Path:
    """Short YOLO26n-seg baseline on the ROI crops. Mosaic stays off."""
    data_yaml = build_day3_bundle(cfg)
    weights = _pretrained(cfg)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    device: int | str = int(cfg.device) if str(cfg.device).isdigit() else cfg.device
    logger.info(
        "Day 3 baseline %s for %s epochs, mosaic %s, degrees %s, scale %s",
        weights.name,
        cfg.day3_baseline_epochs,
        cfg.day3_mosaic,
        cfg.day3_degrees,
        cfg.day3_scale,
    )
    model.train(
        data=str(data_yaml),
        epochs=cfg.day3_baseline_epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=device,
        project=str(cfg.output_dir / "runs"),
        name="day3-seg",
        exist_ok=True,
        seed=cfg.seed,
        workers=cfg.yolo_workers,
        patience=cfg.day3_baseline_epochs,
        mosaic=cfg.day3_mosaic,
        close_mosaic=0,
        degrees=cfg.day3_degrees,
        scale=cfg.day3_scale,
        flipud=cfg.yolo_flipud,
        fliplr=cfg.yolo_fliplr,
        hsv_h=cfg.yolo_hsv_h,
        hsv_s=cfg.yolo_hsv_s,
        hsv_v=cfg.yolo_hsv_v,
        copy_paste=0.0,
        plots=True,
    )
    best = day3_weights(cfg)
    if not best.is_file():
        raise ConfigError(f"Training finished without {best}.")
    logger.info("Day 3 weights: %s", best)
    return best


def train_s5(cfg: AppConfig) -> Path:
    """Retrain on real ROI crops plus synthetic piles. Mosaic stays off for this one change."""
    from coffee.data.piles import build_piles

    build_piles(cfg)
    data_yaml = cfg.output_dir / "day3" / "yolo_s5" / "data.yaml"
    if not data_yaml.is_file():
        raise ConfigError(f"Missing {data_yaml}.")
    weights = _pretrained(cfg)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    device: int | str = int(cfg.device) if str(cfg.device).isdigit() else cfg.device
    logger.info(
        "S5 train %s for %s epochs at imgsz %s. Mosaic stays 0. Real val images are held out.",
        weights.name,
        cfg.day3_s5_epochs,
        cfg.imgsz,
    )
    model.train(
        data=str(data_yaml),
        epochs=cfg.day3_s5_epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=device,
        project=str(cfg.output_dir / "runs"),
        name="day3-s5",
        exist_ok=True,
        seed=cfg.seed,
        workers=cfg.yolo_workers,
        patience=cfg.day3_s5_epochs,
        mosaic=0.0,
        close_mosaic=0,
        degrees=cfg.day3_degrees,
        scale=cfg.day3_scale,
        flipud=cfg.yolo_flipud,
        fliplr=cfg.yolo_fliplr,
        hsv_h=0.0,
        hsv_s=cfg.yolo_hsv_s,
        hsv_v=cfg.yolo_hsv_v,
        copy_paste=0.0,
        plots=True,
    )
    best = s5_weights(cfg)
    if not best.is_file():
        raise ConfigError(f"Training finished without {best}.")
    logger.info("S5 weights: %s", best)
    return best


def day3_weights(cfg: AppConfig) -> Path:
    return cfg.output_dir / "runs" / "day3-seg" / "weights" / "best.pt"


def s5_weights(cfg: AppConfig) -> Path:
    return cfg.output_dir / "runs" / "day3-s5" / "weights" / "best.pt"


def _polygons_in_roi(
    class_ids: list[int],
    polygons: list[np.ndarray],
    roi: Roi,
) -> tuple[list[int], list[np.ndarray], int]:
    kept_ids: list[int] = []
    kept_polygons: list[np.ndarray] = []
    dropped = 0
    for class_id, polygon in zip(class_ids, polygons):
        shifted = shift_polygon(polygon, roi)
        if shifted is None:
            dropped += 1
            continue
        kept_ids.append(class_id)
        kept_polygons.append(shifted)
    return kept_ids, kept_polygons, dropped


def train_yolo(cfg: AppConfig) -> Path:
    data_yaml = build_train_bundle(cfg)
    weights = _pretrained(cfg)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    device: int | str = int(cfg.device) if str(cfg.device).isdigit() else cfg.device
    logger.info("Training %s for up to %s epochs on %s", weights.name, cfg.epochs, device)
    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=device,
        project=str(cfg.output_dir / "runs"),
        name="yolo26n-seg",
        exist_ok=True,
        seed=cfg.seed,
        workers=cfg.yolo_workers,
        patience=cfg.yolo_patience,
        mosaic=cfg.yolo_mosaic,
        close_mosaic=cfg.yolo_close_mosaic,
        degrees=cfg.yolo_degrees,
        scale=cfg.yolo_scale,
        flipud=cfg.yolo_flipud,
        fliplr=cfg.yolo_fliplr,
        hsv_h=cfg.yolo_hsv_h,
        hsv_s=cfg.yolo_hsv_s,
        hsv_v=cfg.yolo_hsv_v,
        copy_paste=cfg.yolo_copy_paste,
        plots=True,
    )
    best = cfg.output_dir / "runs" / "yolo26n-seg" / "weights" / "best.pt"
    if not best.is_file():
        raise ConfigError(f"Training finished without {best}.")
    logger.info("Best weights: %s", best)
    return best


def export_ncnn(cfg: AppConfig, weights: Path | None = None) -> Path:
    best = weights or (cfg.output_dir / "runs" / "yolo26n-seg" / "weights" / "best.pt")
    if not best.is_file():
        raise ConfigError(f"Missing weights {best}. Run train-yolo first.")
    from ultralytics import YOLO

    model = YOLO(str(best))
    device: int | str = int(cfg.device) if str(cfg.device).isdigit() else cfg.device
    logger.info("Exporting %s to NCNN at imgsz %s", best, cfg.imgsz)
    exported = Path(model.export(format="ncnn", imgsz=cfg.imgsz, device=device))
    destination = cfg.model_dir / "yolo26n-seg_ncnn"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(exported, destination)
    logger.info("NCNN export: %s", destination)
    return destination


def _pretrained(cfg: AppConfig) -> Path:
    destination = cfg.model_dir / cfg.yolo_weights_name
    if destination.is_file():
        return destination
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    logger.info("Downloading %s", cfg.yolo_weights_name)
    YOLO(cfg.yolo_weights_name)
    downloaded = Path(cfg.yolo_weights_name)
    if not downloaded.is_file():
        raise ConfigError(f"Download did not produce {cfg.yolo_weights_name}.")
    shutil.copy2(downloaded, destination)
    return destination


def _copy_pair(image_dir: Path, label_dir: Path, image_dest: Path, label_dest: Path, prefix: str) -> None:
    if not image_dir.is_dir():
        return
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ConfigError(f"Image has no label: {image_path}")
        shutil.copy2(image_path, image_dest / f"{prefix}{image_path.name}")
        shutil.copy2(label_path, label_dest / f"{prefix}{image_path.stem}.txt")
