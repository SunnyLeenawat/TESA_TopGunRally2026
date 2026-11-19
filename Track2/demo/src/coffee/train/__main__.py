"""python -m coffee.train yolo|export|ae|ae-day3|day3|s5"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from coffee.config import ConfigError, load_config
from coffee.train.ae import train_autoencoder, train_day3_autoencoder
from coffee.infer.ncnn_seg import inspect_export
from coffee.train.yolo import export_ncnn, train_day3, train_s5, train_yolo

logger = logging.getLogger("coffee.train")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(prog="python -m coffee.train")
    parser.add_argument("command", choices=("yolo", "export", "ae", "ae-day3", "day3", "s5"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, output_dir=args.output)
        if args.command == "yolo":
            train_yolo(cfg)
        elif args.command == "ae":
            train_autoencoder(cfg)
        elif args.command == "ae-day3":
            train_day3_autoencoder(cfg)
        elif args.command == "day3":
            weights = train_day3(cfg)
            export_dir = export_ncnn(cfg, weights)
            inspect_export(export_dir, cfg)
        elif args.command == "s5":
            weights = train_s5(cfg)
            export_dir = export_ncnn(cfg, weights)
            inspect_export(export_dir, cfg)
        else:
            export_ncnn(cfg)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
