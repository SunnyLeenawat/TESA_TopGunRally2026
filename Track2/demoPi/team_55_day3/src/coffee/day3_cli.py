"""Day 3 root commands. infer.py and client.py are the contest entries."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from coffee.client.run import run_client
from coffee.config import ConfigError, load_config
from coffee.infer.contract import run_infer

logger = logging.getLogger("coffee.day3")


def _logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
            stream=sys.stdout,
        )


def infer_main(argv: list[str] | None = None) -> int:
    """python infer.py --input <folder> --weights <folder> --out <folder>."""
    _logging()
    parser = argparse.ArgumentParser(prog="infer.py")
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder.")
    parser.add_argument("--weights", type=Path, required=True, help="Folder of NCNN weights.")
    parser.add_argument("--out", type=Path, required=True, help="Output folder.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.input.exists():
        logger.error("Input not found: %s", args.input)
        return 2
    if not args.weights.is_dir():
        logger.error("Weights path is not a folder: %s", args.weights)
        return 2
    try:
        cfg = load_config(args.config)
        ok = run_infer(cfg, args.input, args.weights, args.out)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    return 0 if ok else 1


def train_main(argv: list[str] | None = None) -> int:
    """Host training entry. python train.py --run starts the Day 3 trainer."""
    _logging()
    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run", action="store_true", help="Train on the host. Needs torch and ultralytics.")
    args = parser.parse_args(argv)
    if not args.run:
        logger.error("Training runs on the host: python train.py --run")
        return 2
    try:
        import ultralytics  # noqa: F401
    except ImportError as exc:
        logger.error("ultralytics is not installed. Use the train container. %s", exc)
        return 2
    from coffee.train.__main__ import main as package_main

    command = ["s5"]
    if args.config is not None:
        command.extend(["--config", str(args.config)])
    return package_main(command)


def client_main(argv: list[str] | None = None) -> int:
    """Fetch images, infer, upload results, and publish MQTT."""
    _logging()
    parser = argparse.ArgumentParser(prog="client.py")
    parser.add_argument("--weights", type=Path, default=None, help="Folder of NCNN weights.")
    parser.add_argument("--out", type=Path, default=None, help="Working output folder.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many images. Default: until HTTP 403.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    if not cfg.http_base:
        logger.error("COFFEE_HTTP_BASE is empty. Set it or run make day3-mock.")
        return 2
    if args.weights is None or args.out is None:
        logger.error("--weights and --out are required.")
        return 2
    try:
        return run_client(cfg, args.weights, args.out, args.limit)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
