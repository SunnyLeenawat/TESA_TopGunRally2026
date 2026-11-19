from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from coffee.config import load_config
from coffee.data.beans import Instance
from coffee.data.day3_labels import (
    harvest_separated,
    image_folds,
    image_holdout,
    load_string_labels,
    prepare_labeled_set,
)
from coffee.data.labels import read_yolo_seg
from coffee.data.validate import DatasetError

from test_config import _write_config


def test_holdout_and_folds_are_image_level() -> None:
    names = [f"img_{index:02d}.jpg" for index in range(8)]
    train, val = image_holdout(names, 0.25, seed=1)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(names)
    assert len(val) == 2
    folds = image_folds(names, 4, seed=1)
    assert len(folds) == 4
    seen_val: list[str] = []
    for fold_train, fold_val in folds:
        assert set(fold_train).isdisjoint(fold_val)
        assert set(fold_train) | set(fold_val) == set(names)
        seen_val.extend(fold_val)
    assert sorted(seen_val) == sorted(names)


def test_string_labels_become_train_ids(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_image(tmp_path / "data" / "mix" / "a.jpg", 40, 30, 10)
    _write_image(tmp_path / "data" / "mix" / "b.jpg", 40, 30, 80)
    _write_labels(
        tmp_path / "data" / "mix" / "labels.json",
        {
            "a.jpg": [("husk", 1)],
            "b.jpg": [("full_black", 1)],
        },
    )
    destination = prepare_labeled_set(cfg)
    stats = json.loads((destination / "stats.json").read_text(encoding="utf-8"))
    assert stats["beans"] == 2
    assert stats["per_class"]["husk"] == 1
    assert stats["per_class"]["black"] == 1
    assert set(stats["holdout"]["train"]).isdisjoint(stats["holdout"]["val"])
    folds = json.loads((destination / "folds.json").read_text(encoding="utf-8"))
    assert "optimistic" in folds["leakage_warning"]
    data_yaml = yaml.safe_load((destination / "yolo" / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["names"][0] == "husk"
    assert data_yaml["names"][5] == "black"
    label_dir = destination / "yolo" / "labels"
    written = list(label_dir.rglob("*.txt"))
    assert len(written) == 2
    ids: set[int] = set()
    for path in written:
        class_ids, polygons = read_yolo_seg(path, 40, 30)
        ids.add(class_ids[0])
        assert polygons[0].shape[0] == 4
    assert ids == {0, 5}
    assert (destination / "preview" / "sheet.jpg").is_file()
    assert (destination / "crops" / "labeled" / "husk").is_dir()
    assert (destination / "crops" / "labeled" / "black").is_dir()


def test_integer_class_is_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            [
                {
                    "image_name": "a.jpg",
                    "objects": [{"id": 1, "class": 0, "polygon": [[0, 0], [1, 0], [1, 1]]}],
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="string"):
        load_string_labels(path, cfg)


def test_harvest_skips_mix_copy_and_duplicate_bytes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    classes = cfg.classes
    for index, name in enumerate(classes):
        _write_image(tmp_path / "data" / "dataset_TESA" / name / "one.jpg", 16, 16, index + 1)
    black = (tmp_path / "data" / "dataset_TESA" / "black" / "one.jpg").read_bytes()
    husk = (tmp_path / "data" / "dataset_TESA" / "husk" / "one.jpg").read_bytes()
    duplicate_black = tmp_path / "data" / "train" / "full_black" / "one.jpg"
    duplicate_husk = tmp_path / "data" / "train" / "hull_husk" / "one.jpg"
    duplicate_black.parent.mkdir(parents=True)
    duplicate_husk.parent.mkdir(parents=True)
    duplicate_black.write_bytes(black)
    duplicate_husk.write_bytes(husk)
    _write_image(tmp_path / "data" / "dataset_TESA" / "mix" / "Mixed_grid1.jpg", 16, 16, 200)

    def extract(image: np.ndarray, class_name: str) -> list[Instance]:
        crop = image[:4, :4].copy()
        mask = np.full(crop.shape[:2], 255, np.uint8)
        return [Instance(class_name, np.array([[0, 0], [3, 0], [3, 3]], np.float32), crop, mask)]

    counts = harvest_separated(cfg, extract)
    assert counts == {name: 1 for name in classes}
    assert not (cfg.output_dir / "day3" / "crops" / "separated" / "mix").exists()


def _cfg(tmp_path: Path):
    order = "[husk, parchment, immature, good, shell, black, broken]"
    path = _write_config(tmp_path, classes=order, registry=True)
    return load_config(path, data_dir=tmp_path / "data", output_dir=tmp_path / "out")


def _write_image(path: Path, width: int, height: int, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), value, np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def _write_labels(path: Path, images: dict[str, list[tuple[str, int]]]) -> None:
    polygon = [[2, 2], [20, 2], [20, 12], [2, 12]]
    payload = []
    for image_name, beans in images.items():
        payload.append(
            {
                "image_name": image_name,
                "objects": [
                    {"id": source_id, "class": class_name, "polygon": polygon}
                    for class_name, source_id in beans
                ],
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
