from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from coffee.config import CLASS_NAMES, ConfigError, load_config
from coffee.hello import main

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_coffee_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("COFFEE_"):
            monkeypatch.delenv(key, raising=False)


def test_shipped_config_loads() -> None:
    cfg = load_config()
    assert cfg.day == 1
    assert cfg.classes == CLASS_NAMES
    assert cfg.class_to_id["husk"] == 0
    assert cfg.class_to_id["immature"] == 2
    assert cfg.imgsz == 640
    assert cfg.label_format == "yolo_seg_txt"
    assert cfg.data_dir == (REPO / "data").resolve()
    assert cfg.infer_sources == ("image", "folder")
    assert cfg.ncnn_threads == 4
    assert cfg.extract_backend == "fastsam"
    assert cfg.sample_aliases["fullblack"] == "black"
    assert cfg.sample_aliases["hull"] == "husk"
    assert cfg.sample_aliases["unripe"] == "immature"


def test_env_and_cli_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COFFEE_DATA_DIR", "/tmp/from-env")
    monkeypatch.setenv("COFFEE_DAY", "2")
    monkeypatch.setenv("COFFEE_IMGSZ", "320")
    from_env = load_config()
    assert from_env.data_dir == Path("/tmp/from-env")
    assert from_env.day == 2
    assert from_env.imgsz == 320
    from_cli = load_config(day=3, data_dir=Path("/tmp/from-cli"), output_dir=Path("custom-out"))
    assert from_cli.day == 3
    assert from_cli.data_dir == Path("/tmp/from-cli")
    assert from_cli.output_dir == (REPO / "custom-out").resolve()


def test_rejects_unknown_class_list(tmp_path: Path) -> None:
    path = _write_config(tmp_path, classes="[good, shell]", registry=True)
    with pytest.raises(ConfigError, match="classes must match the registry"):
        load_config(path)


def test_rejects_bad_day_and_imgsz(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="day"):
        load_config(_write_config(tmp_path, day="9"))
    with pytest.raises(ConfigError, match="yolo.imgsz"):
        load_config(_write_config(tmp_path, imgsz="224"))


def test_rejects_other_label_format(tmp_path: Path) -> None:
    path = _write_config(tmp_path, label_format="coco")
    with pytest.raises(ConfigError, match="labels.format"):
        load_config(path)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(tmp_path / "config" / "config.yaml")


def test_hello_logs_overrides(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    code = main(
        [
            "--day",
            "2",
            "--data-dir",
            "/tmp/coffee-override",
            "--output",
            "/tmp/coffee-out",
            "--input",
            "/tmp/pile.jpg",
        ]
    )
    assert code == 0
    text = caplog.text
    assert "day=2" in text
    assert "data_dir=/tmp/coffee-override exists=False" in text
    assert "output_dir=/tmp/coffee-out" in text
    assert "input=/tmp/pile.jpg" in text


def test_hello_returns_2_on_bad_config(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    missing = tmp_path / "config" / "config.yaml"
    caplog.set_level(logging.ERROR)
    code = main(["--config", str(missing)])
    assert code == 2
    assert "Config file not found" in caplog.text


def _write_config(
    tmp_path: Path,
    *,
    day: str = "${COFFEE_DAY:-1}",
    classes: str = "[good, shell, full_black, hull_husk, broken, parchment, immature]",
    imgsz: str = "${COFFEE_IMGSZ:-640}",
    label_format: str = "yolo_seg_txt",
    registry: bool = False,
) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    registry_line = ""
    if registry:
        source = REPO / "config" / "classes.yaml"
        (config_dir / "classes.yaml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        registry_line = "registry: classes.yaml"
    path = config_dir / "config.yaml"
    path.write_text(
        f"""
day: {day}
seed: 1
{registry_line}
paths:
  data_dir: ${{COFFEE_DATA_DIR:-data}}
  output_dir: ${{COFFEE_OUTPUT_DIR:-outputs}}
  model_dir: ${{COFFEE_MODEL_DIR:-models}}
classes: {classes}
dataset:
  train_dirname: train
  test_dirname: test
  image_extensions: [".jpg"]
labels:
  format: {label_format}
yolo:
  weights_name: yolo26n-seg.pt
  imgsz: {imgsz}
  device: ${{COFFEE_DEVICE:-0}}
  epochs: 1
  batch: 1
ncnn:
  threads: 4
  use_vulkan: false
autoencoder:
  enabled: true
  fusion_enabled: false
  crop_size: 64
infer:
  sources: [image, folder]
extract:
  backend: classical
""",
        encoding="utf-8",
    )
    return path
