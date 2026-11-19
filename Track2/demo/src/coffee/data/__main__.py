"""Dataset commands: python -m coffee.data mock|prep|synth|validate|day3|piles."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from coffee.config import ConfigError, load_config
from coffee.data.day3_labels import run_day3
from coffee.data.piles import build_piles
from coffee.data.pipeline import run_mock, run_prep, run_synth
from coffee.data.validate import DatasetError, validate_dataset

logger = logging.getLogger("coffee.data")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(prog="python -m coffee.data")
    parser.add_argument("command", choices=("mock", "prep", "synth", "validate", "day3", "piles"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, data_dir=args.data_dir, output_dir=args.output)
        if args.command == "mock":
            run_mock(cfg)
        elif args.command == "prep":
            run_prep(cfg)
        elif args.command == "synth":
            run_synth(cfg)
        elif args.command == "day3":
            run_day3(cfg)
        elif args.command == "piles":
            build_piles(cfg)
        else:
            root = cfg.data_dir / cfg.mock_dirname
            if not (root / cfg.train_dirname).is_dir():
                root = cfg.data_dir
            validate_dataset(root, cfg).raise_if_problems()
            logger.info("Dataset OK: %s", root)
    except (ConfigError, DatasetError) as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
