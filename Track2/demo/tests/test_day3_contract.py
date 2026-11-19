from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from coffee.config import load_config
from coffee.eval.score import (
    count_ratio,
    greedy_pairs,
    hungarian_pairs,
    latency_points,
    mean_iou,
    score_folder,
)
from coffee.eval.validate import ContractError, validate_outputs
from coffee.infer.contract import run_infer
from coffee.data.roi import Roi
from coffee.infer.instances import paste_crop_mask, resolve_instances, resolve_on_roi
from coffee.infer.ncnn_seg import Detection

from test_config import _write_config


def test_higher_score_wins_the_overlap_and_ids_stay_dense() -> None:
    shape = (40, 40)
    high = np.zeros(shape, np.uint8)
    low = np.zeros(shape, np.uint8)
    high[5:30, 5:25] = 255
    low[5:30, 15:35] = 255
    label, instances = resolve_instances(
        [("shell", 0.2, low), ("good", 0.9, high)],
        shape,
        dedup_iou=0.9,
        min_area=10,
    )
    assert [item.class_name for item in instances] == ["good", "shell"]
    assert [item.id for item in instances] == [1, 2]
    assert int(label[10, 10]) == 1
    assert int(label[10, 20]) == 1
    assert int(label[10, 30]) == 2
    assert instances[0].polygon[0][0] >= 0


def test_duplicate_mask_is_dropped() -> None:
    shape = (30, 30)
    mask = np.zeros(shape, np.uint8)
    mask[5:25, 5:25] = 255
    label, instances = resolve_instances(
        [("good", 0.8, mask), ("good", 0.3, mask.copy())],
        shape,
        dedup_iou=0.5,
        min_area=10,
    )
    assert len(instances) == 1
    assert set(np.unique(label).tolist()) == {0, 1}


def test_roi_resolve_matches_the_full_frame() -> None:
    full = (80, 100)
    roi = Roi(10, 15, 70, 60, "test")
    inside = np.zeros((roi.height, roi.width), np.uint8)
    edge = np.zeros_like(inside)
    inside[2:20, 2:18] = 255
    edge[0:12, roi.width - 10 :] = 255
    candidates = [("shell", 0.2, edge), ("good", 0.9, inside)]
    full_label, full_instances = resolve_instances(
        [(name, score, paste_crop_mask(mask, full, roi)) for name, score, mask in candidates],
        full,
        dedup_iou=0.9,
        min_area=10,
    )
    label, instances = resolve_on_roi(candidates, roi, full, dedup_iou=0.9, min_area=10)
    assert np.array_equal(label, full_label)
    assert [item.polygon for item in instances] == [item.polygon for item in full_instances]
    assert instances[0].polygon[0][0] >= roi.x0


def test_hungarian_beats_the_greedy_trap() -> None:
    score = np.array([[0.6, 0.5], [0.5, 0.1]])
    greedy = greedy_pairs(score)
    hungarian = hungarian_pairs(score)
    assert sum(score[row, col] for row, col in greedy) == pytest.approx(0.7)
    assert {(row, col) for row, col in hungarian} == {(0, 1), (1, 0)}


def test_latency_drops_on_the_first_extra_centisecond() -> None:
    assert latency_points(1.0, 30) == 25
    assert latency_points(1.01, 30) == 20
    assert latency_points(0.4, 0) == 0
    assert mean_iou([1.0], 1, 2) == pytest.approx(0.5)
    assert count_ratio(1, 1, 2) == pytest.approx(0.5)


def test_infer_writes_a_valid_contract_and_continues_after_one_failure(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    first = tmp_path / "data" / "a.jpg"
    second = tmp_path / "data" / "b.jpg"
    _write_gray(first, 200, 180)
    _write_gray(second, 200, 180)
    calls = {"n": 0}

    def predict(crop: np.ndarray) -> list[Detection]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        mask = np.zeros(crop.shape[:2], np.uint8)
        mask[8:40, 8:50] = 255
        return [Detection(3, 0.4, np.array([8, 8, 50, 40], np.float32), mask)]

    ok = run_infer(cfg, tmp_path / "data", tmp_path / "weights", tmp_path / "out", predict)
    assert ok
    payload = json.loads((tmp_path / "out" / "predictions.json").read_text(encoding="utf-8"))
    by_name = {entry["image_name"]: entry for entry in payload}
    assert len(by_name["a.jpg"]["objects"]) == 1
    assert by_name["a.jpg"]["objects"][0]["class"] == "good"
    assert "score" not in by_name["a.jpg"]["objects"][0]
    assert by_name["b.jpg"]["objects"] == []
    message = json.loads((tmp_path / "out" / "mqtt" / "b.json").read_text(encoding="utf-8"))
    assert message["counts"]["good"] == 0
    assert message["defect_ratio"] == 0.0
    mask = cv2.imread(str(tmp_path / "out" / "seg_mask" / "a.png"), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (180, 200)
    assert int(mask.max()) == 1


def test_perfect_prediction_scores_full_iou_and_count(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    image_path = tmp_path / "images" / "bean.jpg"
    _write_gray(image_path, 80, 60)
    polygon = [[10, 10], [40, 10], [40, 30], [10, 30]]
    mask = np.zeros((60, 80), np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, np.int32).reshape(-1, 1, 2)], 1)
    out = tmp_path / "out"
    (out / "seg_mask").mkdir(parents=True)
    (out / "mqtt").mkdir()
    (out / "vis").mkdir()
    cv2.imwrite(str(out / "seg_mask" / "bean.png"), mask)
    cv2.imwrite(str(out / "vis" / "bean.png"), np.zeros((60, 80, 3), np.uint8))
    prediction = [{"image_name": "bean.jpg", "objects": [{"id": 1, "class": "good", "polygon": polygon}]}]
    (out / "predictions.json").write_text(json.dumps(prediction), encoding="utf-8")
    counts = {name: 0 for name in cfg.submission_names}
    counts["good"] = 1
    (out / "mqtt" / "bean.json").write_text(
        json.dumps({"image": "bean.jpg", "latency_ms": 500, "counts": counts, "defect_ratio": 0.0}),
        encoding="utf-8",
    )
    validate_outputs(out, [image_path], cfg.submission_names, cfg.good_name)
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps([{"image_name": "bean.jpg", "objects": [{"id": 1, "class": "good", "polygon": polygon}]}]),
        encoding="utf-8",
    )
    report = score_folder(cfg, out, labels, image_path.parent)
    assert report.payload["p1_agnostic"] == pytest.approx(15)
    assert report.payload["p2_agnostic"] == pytest.approx(15)
    assert report.payload["latency_agnostic"] == pytest.approx(25)
    assert report.payload["mqtt"] == pytest.approx(5)
    assert report.payload["projected_agnostic"] == pytest.approx(60)


def test_validate_rejects_a_count_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    image_path = tmp_path / "images" / "bean.jpg"
    _write_gray(image_path, 40, 40)
    out = tmp_path / "out"
    (out / "seg_mask").mkdir(parents=True)
    (out / "mqtt").mkdir()
    (out / "vis").mkdir()
    cv2.imwrite(str(out / "seg_mask" / "bean.png"), np.zeros((40, 40), np.uint8))
    cv2.imwrite(str(out / "vis" / "bean.png"), np.zeros((40, 40, 3), np.uint8))
    (out / "predictions.json").write_text(
        json.dumps([{"image_name": "bean.jpg", "objects": []}]),
        encoding="utf-8",
    )
    counts = {name: 1 if name == "good" else 0 for name in cfg.submission_names}
    (out / "mqtt" / "bean.json").write_text(
        json.dumps({"image": "bean.jpg", "latency_ms": 1, "counts": counts, "defect_ratio": 0.0}),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="counts disagree"):
        validate_outputs(out, [image_path], cfg.submission_names, cfg.good_name)


def _cfg(tmp_path: Path):
    order = "[husk, parchment, immature, good, shell, black, broken]"
    path = _write_config(tmp_path, classes=order, registry=True)
    return load_config(path, data_dir=tmp_path / "data", output_dir=tmp_path / "unused")


def _write_gray(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), 210, np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")
