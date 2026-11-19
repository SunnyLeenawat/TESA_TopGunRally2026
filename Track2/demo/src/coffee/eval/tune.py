"""Sweep confidence, dedup, and the optional mask gate. Report score against latency."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

from coffee.config import ConfigError, load_config
from coffee.eval.score import score_folder
from coffee.infer.contract import run_infer

logger = logging.getLogger("coffee.eval.tune")

_CONFS = (0.01, 0.02, 0.05, 0.08, 0.12)
_DEDUPS = (0.3, 0.5, 0.7)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    del argv
    try:
        cfg = load_config()
        rows = sweep(cfg)
    except (ConfigError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 2
    destination = cfg.output_dir / "day3" / "tune.json"
    destination.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        logger.info(
            "%s aware=%.2f agnostic=%.2f median=%.3fs iou_aware=%.3f count_aware=%.3f",
            row["name"],
            row["projected_aware"],
            row["projected_agnostic"],
            row["median_s"],
            row["iou_aware"],
            row["count_aware"],
        )
    logger.info("Wrote %s", destination)
    return 0


def sweep(cfg) -> list[dict[str, object]]:
    labels = cfg.data_dir / cfg.mixed_dirname / cfg.labels_filename
    weights = cfg.model_dir / "yolo26n-seg_ncnn"
    if not (weights / "model.ncnn.param").is_file():
        raise ConfigError(f"Missing NCNN export {weights}. Train s5 first.")
    val_dir = _holdout_dir(cfg)
    rows: list[dict[str, object]] = []
    for conf in _CONFS:
        rows.append(_measure(cfg, val_dir, weights, labels, f"val conf {conf}", conf, cfg.day3_dedup_iou, False))
    best = max(rows, key=lambda row: float(row["projected_aware"]))
    best_conf = float(best["conf"])
    dedup_rows = []
    for dedup in _DEDUPS:
        if abs(dedup - cfg.day3_dedup_iou) < 1e-6:
            dedup_rows.append(best)
            continue
        dedup_rows.append(_measure(cfg, val_dir, weights, labels, f"val dedup {dedup}", best_conf, dedup, False))
    best = max([*rows, *dedup_rows], key=lambda row: float(row["projected_aware"]))
    refine_row = _measure(
        cfg, val_dir, weights, labels, "val refine", float(best["conf"]), float(best["dedup"]), True
    )
    chosen = refine_row if _refine_wins(best, refine_row) else best
    full = _measure(
        cfg,
        cfg.data_dir / cfg.mixed_dirname,
        weights,
        labels,
        "all16 " + str(chosen["name"]),
        float(chosen["conf"]),
        float(chosen["dedup"]),
        bool(chosen["refine"]),
    )
    return [*rows, *dedup_rows, refine_row, full]


def _refine_wins(current: dict[str, object], refine_row: dict[str, object]) -> bool:
    gain = float(refine_row["projected_aware"]) - float(current["projected_aware"])
    slower = float(refine_row["median_s"]) - float(current["median_s"])
    return gain > 0.3 and slower < 0.15


def _measure(cfg, image_dir: Path, weights: Path, labels: Path, name: str, conf: float, dedup: float, refine: bool) -> dict[str, object]:
    safe = name.replace(" ", "_").replace(".", "p")
    out_dir = cfg.output_dir / "day3" / "tune" / safe
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ok = run_infer(
        cfg,
        image_dir,
        weights,
        out_dir,
        conf=conf,
        dedup_iou=dedup,
        refine=refine,
        write_vis=False,
    )
    if not ok:
        raise ConfigError(f"Inference contract failed for {name}.")
    report = score_folder(cfg, out_dir, labels, image_dir)
    payload = report.payload
    return {
        "name": name,
        "conf": conf,
        "dedup": dedup,
        "refine": refine,
        "projected_aware": payload["projected_aware"],
        "projected_agnostic": payload["projected_agnostic"],
        "median_s": payload["median_s"],
        "iou_aware": payload["iou_hungarian_aware"],
        "iou_agnostic": payload["iou_hungarian_agnostic"],
        "count_aware": payload["count_aware"],
        "count_agnostic": payload["count_agnostic"],
        "latency_aware": payload["latency_aware"],
        "mqtt": payload["mqtt"],
    }


def _holdout_dir(cfg) -> Path:
    folds_path = cfg.output_dir / "day3" / "folds.json"
    if not folds_path.is_file():
        raise ConfigError(f"Missing {folds_path}.")
    names = json.loads(folds_path.read_text(encoding="utf-8"))["holdout"]["val"]
    destination = cfg.output_dir / "day3" / "tune_val"
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = destination / name
        source = cfg.data_dir / cfg.mixed_dirname / name
        if not target.is_file():
            shutil.copy2(source, target)
    return destination


if __name__ == "__main__":
    raise SystemExit(main())
