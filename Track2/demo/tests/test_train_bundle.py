from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from coffee.config import load_config
from coffee.train.yolo import build_train_bundle

from test_config import _write_config


def test_bundle_puts_synth_in_train_only(tmp_path: Path) -> None:
    output = tmp_path / "out"
    _touch_split(output / "yolo", "train", "kept")
    _touch_split(output / "yolo", "val", "held")
    synth = output / "synth" / "images"
    synth.mkdir(parents=True)
    _write_pair(synth, "pile")
    cfg = load_config(_write_config(tmp_path / "cfg"), output_dir=output)
    data_yaml = build_train_bundle(cfg)
    train_names = {path.name for path in (output / "yolo_bundle" / "images" / "train").iterdir()}
    val_names = {path.name for path in (output / "yolo_bundle" / "images" / "val").iterdir()}
    assert "kept.jpg" in train_names
    assert "synth__pile.jpg" in train_names
    assert val_names == {"held.jpg"}
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert payload["names"][0] == "good"
    assert (output / "yolo_bundle" / "labels" / "train" / "synth__pile.txt").is_file()


def _touch_split(root: Path, split: str, stem: str) -> None:
    (root / "images" / split).mkdir(parents=True)
    (root / "labels" / split).mkdir(parents=True)
    _write_pair(root / "images" / split, stem)
    (root / "images" / split / f"{stem}.txt").replace(root / "labels" / split / f"{stem}.txt")
    (root / "data.yaml").write_text("names:\n  0: good\n", encoding="utf-8")


def _write_pair(folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{stem}.jpg"), np.zeros((8, 8, 3), np.uint8))
    (folder / f"{stem}.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
