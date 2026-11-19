"""YOLO alone versus YOLO plus the encoder, scored on the class-aware holdout.

Fusion stays off unless the aware score rises by at least 0.5 and the median
grows by no more than 50 ms. The config switch is not edited here.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys

from coffee.ae.bank import load_bank
from coffee.ae.crops import masked_square
from coffee.ae.fuse import fuse
from coffee.config import ConfigError, load_config
from coffee.eval.score import score_folder
from coffee.eval.tune import _holdout_dir
from coffee.infer.contract import run_infer
from coffee.infer.ncnn_encoder import NcnnEncoder
from coffee.infer.ncnn_seg import Detection, NcnnSegmenter

logger = logging.getLogger("coffee.eval.ae")

_GAIN = 0.5
_LATENCY_S = 0.05


def keep_fusion(yolo_aware: float, fused_aware: float, yolo_s: float, fused_s: float) -> bool:
    return (fused_aware - yolo_aware) >= _GAIN and (fused_s - yolo_s) <= _LATENCY_S


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    del argv
    try:
        cfg = load_config()
        report = run_day3_ablation(cfg)
    except (ConfigError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 2
    logger.info(
        "YOLO aware %.2f at %.3f s. YOLO+AE aware %.2f at %.3f s. Keep fusion: %s.",
        report["yolo_aware"],
        report["yolo_median_s"],
        report["fused_aware"],
        report["fused_median_s"],
        report["keep_fusion"],
    )
    if not report["keep_fusion"]:
        logger.info("Fusion stays off. The encoder did not clear the score and latency guard.")
    return 0


def run_day3_ablation(cfg) -> dict[str, object]:
    labels = cfg.data_dir / cfg.mixed_dirname / cfg.labels_filename
    weights = cfg.model_dir / "yolo26n-seg_ncnn"
    holdout = _holdout_dir(cfg)
    yolo = _score(cfg, holdout, weights, labels, "yolo", None)
    fused_predict = _fused_predict(cfg, weights)
    fused = _score(cfg, holdout, weights, labels, "yolo_ae", fused_predict)
    decision = keep_fusion(
        float(yolo["projected_aware"]),
        float(fused["projected_aware"]),
        float(yolo["median_s"]),
        float(fused["median_s"]),
    )
    report = {
        "yolo_aware": yolo["projected_aware"],
        "yolo_agnostic": yolo["projected_agnostic"],
        "yolo_median_s": yolo["median_s"],
        "fused_aware": fused["projected_aware"],
        "fused_agnostic": fused["projected_agnostic"],
        "fused_median_s": fused["median_s"],
        "keep_fusion": decision,
        "fusion_enabled": False,
        "guard": {"min_aware_gain": _GAIN, "max_extra_seconds": _LATENCY_S},
    }
    path = cfg.output_dir / "ae" / "day3_ablation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)
    return report


def _score(cfg, holdout, weights, labels, name: str, predict) -> dict[str, object]:
    out_dir = cfg.output_dir / "ae" / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ok = run_infer(cfg, holdout, weights, out_dir, predict, write_vis=False)
    if not ok:
        raise ConfigError(f"Contract failed for the {name} ablation pass.")
    payload = score_folder(cfg, out_dir, labels, holdout).payload
    return {
        "projected_aware": payload["projected_aware"],
        "projected_agnostic": payload["projected_agnostic"],
        "median_s": payload["median_s"],
    }


def _fused_predict(cfg, weights):
    segmenter = NcnnSegmenter(
        weights,
        cfg.classes,
        cfg.imgsz,
        cfg.ncnn_threads,
        cfg.day3_infer_conf,
        cfg.yolo_iou,
        cfg.ncnn_use_vulkan,
    )
    encoder = NcnnEncoder(cfg.model_dir / "ae_encoder_ncnn", cfg.ncnn_threads, cfg.ncnn_use_vulkan)
    bank = load_bank(cfg.model_dir / "ae_bank.npz")
    if bank.class_names != cfg.classes:
        raise ConfigError(f"Encoder bank classes {bank.class_names} do not match {cfg.classes}. Retrain the autoencoder.")

    def predict(crop):
        detections = segmenter.predict(crop, conf=cfg.day3_infer_conf)
        fused: list[Detection] = []
        for detection in detections:
            class_id = detection.class_id
            if detection.score < cfg.ae_low_conf:
                try:
                    square = masked_square(crop, detection.mask, detection.box_xyxy, cfg.ae_crop_size)
                    result = fuse(
                        detection.class_id,
                        float(detection.score),
                        encoder.embed(square),
                        bank,
                        enabled=True,
                        k=cfg.ae_knn_k,
                        knn_share=cfg.ae_knn_share,
                        low_conf=cfg.ae_low_conf,
                        confusable=cfg.ae_confusable,
                    )
                    class_id = result.class_id
                except (ConfigError, ValueError) as exc:
                    logger.warning("Encoder skipped one bean: %s", exc)
            fused.append(
                Detection(class_id, float(detection.score), detection.box_xyxy, detection.mask, detection.input_mask)
            )
        return fused

    return predict


if __name__ == "__main__":
    raise SystemExit(main())
