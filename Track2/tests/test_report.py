from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from coffee.config import load_config
from coffee.infer.ncnn_seg import Detection
from coffee.infer.report import count_classes, defect_ratio, draw_beans, list_inputs, run_day, vote
from coffee.infer.report import Bean

from test_config import _write_config


def test_vote_breaks_ties_by_score_then_lower_id() -> None:
    beans = [
        _bean(1, "shell", 0.9),
        _bean(0, "good", 0.4),
    ]
    assert vote(beans) == (1, "shell")
    tied = [_bean(1, "shell", 0.2), _bean(0, "good", 0.2)]
    assert vote(tied) == (0, "good")
    assert vote(()) == (-1, "none")


def test_defect_ratio_is_non_good_over_total() -> None:
    counts = {"good": 3, "shell": 1, "broken": 0}
    assert defect_ratio(counts) == 0.25
    assert defect_ratio({"good": 0, "shell": 0}) == 0.0


def test_day1_csv_and_day3_json(tmp_path: Path) -> None:
    cfg = load_config(_write_config(tmp_path / "cfg"))
    image_path = tmp_path / "shot.jpg"
    cv2.imwrite(str(image_path), np.zeros((16, 16, 3), np.uint8))
    mask = np.zeros((16, 16), np.uint8)
    mask[2:8, 2:8] = 255

    def predict(_image):
        return [
            Detection(0, 0.8, np.array([2, 2, 8, 8], np.float32), mask),
            Detection(1, 0.3, np.array([1, 1, 6, 6], np.float32), mask),
        ]

    day1 = run_day(cfg, [image_path], tmp_path / "day1", predict)
    text = (tmp_path / "day1" / "predictions.csv").read_text(encoding="utf-8")
    assert "class_name" in text
    assert "good" in text
    assert day1[0].class_name == "good"

    cfg3 = load_config(_write_config(tmp_path / "cfg3"), day=3)
    run_day(cfg3, [image_path], tmp_path / "day3", predict)
    payload = json.loads((tmp_path / "day3" / "predictions.json").read_text(encoding="utf-8"))
    image = payload["images"][0]
    assert image["counts"]["good"] == 1
    assert image["counts"]["shell"] == 1
    assert image["total"] == 2
    assert image["defect_ratio"] == 0.5
    assert "latency_ms" in image
    assert len(image["per_bean"]) == 2
    overlay = cv2.imread(str(tmp_path / "day3" / "overlays" / "shot.jpg"))
    assert overlay.shape == (16, 16, 3)
    assert int(overlay.sum()) > 0


def test_list_inputs_reads_class_folders_and_flat_files(tmp_path: Path) -> None:
    folder = tmp_path / "test"
    (folder / "good").mkdir(parents=True)
    (folder / "good" / "pile.jpg").write_bytes(b"")
    (folder / "notes").mkdir()
    (folder / "notes" / "ignore.jpg").write_bytes(b"")
    (folder / "loose.jpg").write_bytes(b"")
    found = list_inputs(folder, (".jpg",), ("good", "shell"))
    assert [path.name for path in found] == ["pile.jpg", "loose.jpg"]


def test_overlay_paints_the_mask() -> None:
    image = np.zeros((12, 12, 3), np.uint8)
    mask = np.zeros((12, 12), np.uint8)
    mask[4:9, 4:9] = 255
    bean = Bean(0, "good", 0.5, (4, 4, 9, 9), None, mask)
    painted = draw_beans(image, [bean])
    assert int(painted[6, 6].sum()) > 0
    assert int(painted[0, 0].sum()) == 0


def test_counts_keep_every_class() -> None:
    counts = count_classes([_bean(0, "good", 1.0)], ("good", "shell"))
    assert counts == {"good": 1, "shell": 0}


def _bean(class_id: int, name: str, score: float) -> Bean:
    return Bean(class_id, name, score, (0, 0, 1, 1), None, np.zeros((2, 2), np.uint8))
