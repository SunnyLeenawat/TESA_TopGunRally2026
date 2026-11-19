"""python -m coffee.eval validate|score"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from coffee.config import ConfigError, load_config
from coffee.eval.score import score_folder
from coffee.eval.validate import ContractError, validate_outputs
from coffee.infer.contract import list_images

logger = logging.getLogger("coffee.eval")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(prog="python -m coffee.eval")
    parser.add_argument("command", choices=("validate", "score"))
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        images = list_images(args.images, cfg.image_extensions)
        if args.command == "validate":
            validate_outputs(args.pred, images, cfg.submission_names, cfg.good_name)
            logger.info("Output contract passed for %s image(s).", len(images))
            return 0
        if args.labels is None:
            raise ConfigError("score requires --labels.")
        report = score_folder(cfg, args.pred, args.labels, args.images)
    except (ConfigError, ContractError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2
    payload = report.payload
    for row in payload["images"]:
        logger.info(
            "%s gt=%s pred=%s iou=%.3f count=%.3f",
            row["image"],
            row["n_gt"],
            row["n_pred"],
            row["iou_hungarian_agnostic"],
            row["count_agnostic"],
        )
    logger.info(
        "Hungarian IoU agnostic %.3f aware %.3f. Greedy IoU agnostic %.3f aware %.3f.",
        payload["iou_hungarian_agnostic"],
        payload["iou_hungarian_aware"],
        payload["iou_greedy_agnostic"],
        payload["iou_greedy_aware"],
    )
    logger.info(
        "Count agnostic %.3f aware %.3f. Median %.3f s. MQTT %.1f.",
        payload["count_agnostic"],
        payload["count_aware"],
        payload["median_s"],
        payload["mqtt"],
    )
    logger.info(
        "Projected agnostic %.2f (IoU %.2f + count %.2f + latency %.2f + mqtt %.2f). Aware %.2f.",
        payload["projected_agnostic"],
        payload["p1_agnostic"],
        payload["p2_agnostic"],
        payload["latency_agnostic"],
        payload["mqtt"],
        payload["projected_aware"],
    )
    logger.warning("%s", payload["leakage_warning"])
    destination = args.pred / "score.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
