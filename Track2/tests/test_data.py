from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from coffee.config import load_config
from coffee.data.beans import mask_to_instance
from coffee.data.extract import Extractor
from coffee.data.labels import read_yolo_seg, write_yolo_seg
from coffee.data.pipeline import _split_group, run_prep
from coffee.data.synth import render_neat
from coffee.data.validate import DatasetError, validate_dataset

from test_config import _write_config


def test_yolo_seg_roundtrip(tmp_path: Path) -> None:
    polygon = np.asarray([[10, 10], [40, 12], [30, 50]], dtype=np.float32)
    path = tmp_path / "bean.txt"
    write_yolo_seg(path, [4], [polygon], width=100, height=80)
    class_ids, polygons = read_yolo_seg(path, 100, 80)
    assert class_ids == [4]
    assert polygons[0].shape == (3, 2)
    assert polygons[0][0, 0] == pytest.approx(10, abs=0.1)


def test_classical_extractor_finds_separated_beans(tmp_path: Path) -> None:
    image = np.full((300, 400, 3), (40, 90, 160), np.uint8)
    centers = [(80, 70), (220, 70), (80, 200), (240, 210)]
    for cx, cy in centers:
        cv2.ellipse(image, (cx, cy), (28, 18), 20, 0, 360, (40, 180, 40), -1)
    cfg = load_config(_write_config(tmp_path))
    found = Extractor(cfg).extract(image, "good")
    assert len(found) == 4
    assert {item.class_name for item in found} == {"good"}


def test_validator_lists_missing_and_unknown(tmp_path: Path) -> None:
    root = tmp_path / "set"
    (root / "train" / "good").mkdir(parents=True)
    (root / "train" / "not_a_bean").mkdir()
    (root / "test").mkdir()
    cfg = load_config(_write_config(tmp_path / "cfg"))
    report = validate_dataset(root, cfg)
    text = "\n".join(report.problems)
    assert "Unknown class folder" in text
    assert "Missing train class folder" in text
    assert "Empty test folder" in text
    with pytest.raises(DatasetError, match="not usable"):
        report.raise_if_problems()


def test_split_keeps_each_image_intact() -> None:
    paths = [Path(f"img_{index}.jpg") for index in range(4)]
    train, val = _split_group(paths, 0.34, np.random.default_rng(1))
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(paths)
    assert len(val) == 1


def test_prep_splits_by_source_image(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg")
    data = tmp_path / "data"
    output = tmp_path / "out"
    _write_tiny_dataset(data)
    cfg = load_config(cfg_path, data_dir=data, output_dir=output)
    yolo_root = run_prep(cfg)
    train_images = list((yolo_root / "images" / "train").glob("*.jpg"))
    val_images = list((yolo_root / "images" / "val").glob("*.jpg"))
    assert len(train_images) + len(val_images) == 7 * 3
    assert val_images
    for image_path in train_images + val_images:
        label = (yolo_root / "labels" / image_path.parent.name / f"{image_path.stem}.txt")
        class_ids, polygons = read_yolo_seg(label, 80, 60)
        assert class_ids
        assert polygons[0].shape[1] == 2
    assert (output / "crops" / "shell").is_dir()


def test_neat_scene_writes_matching_labels(tmp_path: Path) -> None:
    cfg = load_config(_write_config(tmp_path / "cfg"), output_dir=tmp_path / "out")
    crop = mask_to_instance(_bean_card(), _bean_card_mask(), "shell", min_area_frac=0.01, max_area_frac=0.9, max_aspect=3)
    assert crop is not None
    background = np.full((200, 280, 3), (40, 90, 160), np.uint8)
    scene, placed = render_neat(background, [crop], "shell", "grid3x3", np.random.default_rng(0))
    assert len(placed) == 9
    assert {item.class_name for item in placed} == {"shell"}
    assert scene.shape[:2] == background.shape[:2]


def _bean_card() -> np.ndarray:
    image = np.full((80, 80, 3), (40, 90, 160), np.uint8)
    cv2.ellipse(image, (40, 40), (24, 16), 0, 0, 360, (40, 180, 40), -1)
    return image


def _bean_card_mask() -> np.ndarray:
    mask = np.zeros((80, 80), np.uint8)
    cv2.ellipse(mask, (40, 40), (24, 16), 0, 0, 360, 255, -1)
    return mask


def _write_tiny_dataset(root: Path) -> None:
    classes = ["good", "shell", "full_black", "hull_husk", "broken", "parchment", "immature"]
    polygon = np.asarray([[20, 15], [55, 18], [40, 48]], dtype=np.float32)
    for class_index, name in enumerate(classes):
        folder = root / "train" / name
        folder.mkdir(parents=True)
        for index in range(3):
            image = np.full((60, 80, 3), 20, np.uint8)
            path = folder / f"bean_{index}.jpg"
            cv2.imwrite(str(path), image)
            write_yolo_seg(path.with_suffix(".txt"), [class_index], [polygon], 80, 60)
    test = root / "test"
    test.mkdir()
    cv2.imwrite(str(test / "pile.jpg"), np.full((60, 80, 3), 20, np.uint8))
