"""Print the resolved configuration. This is the Phase 0 path-override check."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from coffee.config import ConfigError, load_config

logger = logging.getLogger("coffee.hello")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coffee.hello",
        description="Show the resolved coffee-edge configuration.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--input", type=Path, default=None, help="Image file or folder (logged only)")
    parser.add_argument("--output", type=Path, default=None, help="Override paths.output_dir")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override paths.data_dir")
    parser.add_argument("--day", type=int, choices=(1, 2, 3), default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(
            args.config,
            day=args.day,
            data_dir=args.data_dir,
            output_dir=args.output,
        )
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    logger.info("config=%s", cfg.config_path)
    logger.info("repo_root=%s", cfg.repo_root)
    logger.info("day=%s", cfg.day)
    logger.info("seed=%s", cfg.seed)
    logger.info("data_dir=%s exists=%s", cfg.data_dir, cfg.data_dir.is_dir())
    if cfg.data_dir.is_dir():
        entries = sorted(path.name for path in cfg.data_dir.iterdir())
        logger.info("data_dir entries: %s", ", ".join(entries) if entries else "(empty)")
    logger.info("output_dir=%s", cfg.output_dir)
    logger.info("model_dir=%s", cfg.model_dir)
    logger.info("classes=%s", ",".join(cfg.classes))
    logger.info(
        "imgsz=%s train_device=%s ncnn_threads=%s",
        cfg.imgsz,
        cfg.device,
        cfg.ncnn_threads,
    )
    logger.info("label_format=%s", cfg.label_format)
    logger.info("ae_enabled=%s ae_fusion_enabled=%s", cfg.ae_enabled, cfg.ae_fusion_enabled)
    logger.info("input=%s", args.input if args.input is not None else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
