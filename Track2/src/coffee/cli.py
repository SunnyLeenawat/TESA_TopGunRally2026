"""python -m coffee.cli --day 1|2|3 --input image-or-folder --output folder

Day 1 writes predictions.csv (one class per image).
Day 2 writes an overlay with a box and a mask for each bean.
Day 3 writes predictions.json (counts, defect ratio, per bean, latency) and the overlay.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from coffee.config import ConfigError, load_config
from coffee.infer.ncnn_seg import NcnnSegmenter, ncnn_model_dir
from coffee.infer.report import list_inputs, run_day

logger = logging.getLogger("coffee.cli")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(prog="python -m coffee.cli")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--day", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, day=args.day)
        images = list_inputs(args.input, cfg.image_extensions, cfg.classes)
        segmenter = NcnnSegmenter(
            ncnn_model_dir(cfg.model_dir, cfg.output_dir),
            cfg.classes,
            cfg.imgsz,
            cfg.ncnn_threads,
            cfg.yolo_conf,
            cfg.yolo_iou,
            cfg.ncnn_use_vulkan,
        )
        root = args.input if args.input.is_dir() else None
        run_day(cfg, images, args.output, segmenter.predict, root)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
