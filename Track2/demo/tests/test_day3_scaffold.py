from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from coffee.config import ConfigError, load_config
from coffee.day3_cli import client_main, infer_main, train_main

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_coffee_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("COFFEE_"):
            monkeypatch.delenv(key, raising=False)


def test_shipped_registry_order() -> None:
    cfg = load_config()
    assert cfg.classes == (
        "husk",
        "parchment",
        "immature",
        "good",
        "shell",
        "black",
        "broken",
    )
    assert cfg.class_to_id["husk"] == 0
    assert cfg.class_to_id["immature"] == 2
    assert cfg.class_to_id["good"] == 3
    assert cfg.submission_names == (
        "good",
        "shell",
        "black",
        "husk",
        "broken",
        "parchment",
        "immature",
    )
    assert cfg.class_aliases["full_black"] == "black"
    assert cfg.class_aliases["hull_husk"] == "husk"
    assert cfg.good_name == "good"
    assert cfg.mixed_dirname == "mix"
    assert cfg.labels_filename == "labels.json"
    assert cfg.separated_dirnames == ("dataset_TESA", "train")
    assert cfg.team == "team_55"
    assert cfg.day3_fold_count == 4
    assert cfg.day3_val_fraction == 0.25
    assert cfg.yolo_hsv_h == 0.0
    assert "full_black" not in cfg.classes


def test_infer_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as caught:
        infer_main(["--help"])
    assert caught.value.code == 0


def test_infer_missing_input_exits_2(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR)
    weights = tmp_path / "weights"
    weights.mkdir()
    code = infer_main(
        [
            "--input",
            str(tmp_path / "missing"),
            "--weights",
            str(weights),
            "--out",
            str(tmp_path / "out"),
            "--config",
            str(REPO / "config" / "config.yaml"),
        ]
    )
    assert code == 2
    assert "Input not found" in caplog.text


def test_train_and_client_stop_before_their_steps(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR)
    assert train_main(["--config", str(REPO / "config" / "config.yaml")]) == 2
    assert client_main(["--config", str(REPO / "config" / "config.yaml")]) == 2
    assert "python train.py --run" in caplog.text
    assert "COFFEE_HTTP_BASE" in caplog.text


def test_registry_mismatch_is_rejected(tmp_path: Path) -> None:
    from test_config import _write_config

    path = _write_config(tmp_path, classes="[good, shell]", registry=True)
    with pytest.raises(ConfigError, match="classes must match the registry"):
        load_config(path)
