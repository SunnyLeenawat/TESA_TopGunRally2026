from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from coffee.config import load_config
from coffee.data.labels import read_yolo_seg, write_yolo_seg
from coffee.data.roi import estimate_roi, shift_polygon
from coffee.train.yolo import build_day3_bundle

from test_config import _write_config


def test_paper_roi_contains_the_dark_bean_and_is_not_the_full_frame() -> None:
    image = np.full((480, 360, 3), 210, np.uint8)
    cv2.ellipse(image, (180, 240), (36, 22), 15, 0, 360, (30, 50, 40), -1)
    roi = estimate_roi(image)
    assert roi.source == "paper"
    assert roi.x0 < 180 < roi.x1
    assert roi.y0 < 240 < roi.y1
    assert roi.width < 360
    assert roi.height < 480


def test_flat_frame_falls_back_to_centre() -> None:
    image = np.full((200, 220, 3), 190, np.uint8)
    roi = estimate_roi(image, fallback_frac=0.7)
    assert roi.source == "center"
    assert roi.x0 > 0 and roi.y0 > 0
    assert roi.x1 < 220 and roi.y1 < 200


def test_shift_polygon_moves_into_the_crop() -> None:
    roi = estimate_roi(np.full((80, 80, 3), 180, np.uint8))
    polygon = np.array([[roi.x0 + 4, roi.y0 + 4], [roi.x0 + 20, roi.y0 + 5], [roi.x0 + 12, roi.y0 + 24]], np.float32)
    shifted = shift_polygon(polygon, roi)
    assert shifted is not None
    assert shifted[:, 0].min() == 4
    assert shifted[:, 1].min() == 4


def test_day3_bundle_rewrites_labels_in_the_crop(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_split(cfg.output_dir / "day3" / "yolo", "train", "one.jpg")
    _write_split(cfg.output_dir / "day3" / "yolo", "val", "two.jpg")
    data_yaml = build_day3_bundle(cfg)
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert payload["names"][0] == "husk"
    cropped = cv2.imread(str(cfg.output_dir / "day3" / "yolo_roi" / "images" / "train" / "one.jpg"))
    assert cropped is not None
    assert cropped.shape[0] < 480 and cropped.shape[1] < 360
    class_ids, polygons = read_yolo_seg(
        cfg.output_dir / "day3" / "yolo_roi" / "labels" / "train" / "one.txt",
        cropped.shape[1],
        cropped.shape[0],
    )
    assert class_ids == [3]
    assert polygons[0][:, 0].min() >= 0
    assert polygons[0][:, 0].max() < cropped.shape[1]


def _cfg(tmp_path: Path):
    order = "[husk, parchment, immature, good, shell, black, broken]"
    path = _write_config(tmp_path, classes=order, registry=True)
    return load_config(path, data_dir=tmp_path / "data", output_dir=tmp_path / "out")


def _write_split(root: Path, split: str, name: str) -> None:
    image = np.full((480, 360, 3), 210, np.uint8)
    cv2.ellipse(image, (180, 240), (36, 22), 15, 0, 360, (30, 50, 40), -1)
    image_path = root / "images" / split / name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"Could not write {image_path}")
    polygon = np.array([[150, 220], [210, 218], [205, 262], [148, 258]], np.float32)
    write_yolo_seg(root / "labels" / split / "one.txt" if split == "train" else root / "labels" / split / "two.txt", [3], [polygon], 360, 480)
    if split == "train":
        names = {index: name for index, name in enumerate(
            ("husk", "parchment", "immature", "good", "shell", "black", "broken")
        )}
        (root / "data.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
