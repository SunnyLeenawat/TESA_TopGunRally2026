"""Fine-tune YOLO26n-seg and export it. NCNN decoding is a separate step."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import yaml

from coffee.config import AppConfig, ConfigError

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
