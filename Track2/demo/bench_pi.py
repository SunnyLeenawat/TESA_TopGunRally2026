#!/usr/bin/env python3
"""Time Day 3 inference on the Pi. python bench_pi.py --input <folder> --weights <folder> --out <folder>."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from coffee.config import ConfigError, load_config
from coffee.infer.bench import log_platform, median_ms, read_temp, read_vcgencmd, decode_throttled
from coffee.infer.contract import list_images, run_infer
from coffee.infer.ncnn_seg import NcnnSegmenter

logger = logging.getLogger("coffee.bench")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(prog="bench_pi.py")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if not args.weights.is_dir():
            raise ConfigError(f"Weights path is not a folder: {args.weights}")
        images = list_images(args.input, cfg.image_extensions)
        if not images:
            raise ConfigError(f"No images in {args.input}.")
        log_platform(cfg)
        _warmup(cfg, images[0], args.weights)
        ok = run_infer(cfg, args.input, args.weights, args.out)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    values = _latencies(args.out / "stages.json")
    median = median_ms(values)
    logger.info("Measured images %s. Median latency_ms %.1f (%.3f s).", len(values), median, median / 1000.0)
    if median > 1000:
        logger.error("Median is over 1.00 s. The latency term is already short of 25 points on this machine.")
    after, bits = read_vcgencmd()
    if after:
        logger.info("After: %s", after)
    temp = read_temp()
    if temp:
        logger.info("After: %s", temp)
    if bits:
        logger.error("Throttle flags after the pass: %s", ", ".join(decode_throttled(bits)))
    return 0 if ok else 1


def _warmup(cfg, image_path: Path, weights: Path) -> None:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ConfigError(f"Could not read warmup image {image_path}.")
    segmenter = NcnnSegmenter(
        weights,
        cfg.classes,
        cfg.imgsz,
        cfg.ncnn_threads,
        cfg.day3_infer_conf,
        cfg.yolo_iou,
        cfg.ncnn_use_vulkan,
    )
    segmenter.predict(image)
    logger.info("Warmup finished on %s. It is not part of the median.", image_path.name)


def _latencies(path: Path) -> list[int]:
    if not path.is_file():
        raise ConfigError(f"Missing {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(item["latency_ms"]) for item in payload]


if __name__ == "__main__":
    raise SystemExit(main())
