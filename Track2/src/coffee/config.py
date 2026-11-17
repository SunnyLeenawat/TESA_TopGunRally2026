"""Load config/config.yaml. Paths are never hard-coded in callers.

Precedence, highest first: CLI arguments, COFFEE_* environment variables
inside ${VAR:-default} placeholders, then the literal YAML value.
Relative paths resolve from the repository root (the parent of config/).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFUSABLE: tuple[tuple[str, str], ...] = (
    ("good", "immature"),
    ("good", "parchment"),
    ("shell", "broken"),
    ("full_black", "hull_husk"),
    ("broken", "immature"),
)

CLASS_NAMES: tuple[str, ...] = (
    "good",
    "shell",
    "full_black",
    "hull_husk",
    "broken",
    "parchment",
    "immature",
)
ALLOWED_IMGSZ: frozenset[int] = frozenset({320, 480, 640})
ALLOWED_SOURCES: frozenset[str] = frozenset({"image", "folder"})
LABEL_FORMAT: str = "yolo_seg_txt"

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """The config file, an override, or a path is unusable."""


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    repo_root: Path
    day: int
    seed: int
    data_dir: Path
    output_dir: Path
    model_dir: Path
    classes: tuple[str, ...]
    class_to_id: dict[str, int]
    train_dirname: str
    test_dirname: str
    image_extensions: tuple[str, ...]
    label_format: str
    yolo_weights_name: str
    imgsz: int
    device: str
    epochs: int
    batch: int
    yolo_patience: int
    yolo_workers: int
    yolo_mosaic: float
    yolo_close_mosaic: int
    yolo_degrees: float
    yolo_scale: float
    yolo_flipud: float
    yolo_fliplr: float
    yolo_hsv_h: float
    yolo_hsv_s: float
    yolo_hsv_v: float
    yolo_copy_paste: float
    yolo_conf: float
    yolo_iou: float
    ncnn_threads: int
    ncnn_use_vulkan: bool
    ae_enabled: bool
    ae_fusion_enabled: bool
    ae_crop_size: int
    ae_latent_dim: int
    ae_epochs: int
    ae_batch: int
    ae_lr: float
    ae_val_fraction: float
    ae_knn_k: int
    ae_knn_share: float
    ae_low_conf: float
    ae_anomaly_quantile: float
    ae_confusable: tuple[tuple[str, str], ...]
    infer_sources: tuple[str, ...]
    samples_dirname: str
    mock_dirname: str
    day3_dirname: str
    layout_suffix: str
    sample_aliases: dict[str, str]
    extract_backend: str
    extract_weights: str
    extract_imgsz: int
    extract_conf: float
    extract_min_area_frac: float
    extract_max_area_frac: float
    extract_max_aspect: float
    extract_lab_distance: float
    synth_images_per_class: int
    synth_day3_per_pattern: int
    synth_pile_beans_min: int
    synth_pile_beans_max: int
    synth_mixed_classes_min: int
    synth_val_fraction: float
    synth_canvas_width: int
    synth_canvas_height: int

    def class_id(self, name: str) -> int:
        try:
            return self.class_to_id[name]
        except KeyError as exc:
            known = ", ".join(self.classes)
            raise ConfigError(f"Unknown class {name!r}. Expected one of: {known}.") from exc


def default_config_path() -> Path:
    """Find config/config.yaml by walking up from this file, or COFFEE_CONFIG."""
    override = os.environ.get("COFFEE_CONFIG", "").strip()
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "config.yaml"
        if candidate.is_file():
            return candidate
    raise ConfigError(
        "Could not find config/config.yaml. Pass --config or set COFFEE_CONFIG."
    )


def load_config(
    config_path: Path | None = None,
    *,
    day: int | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    model_dir: Path | None = None,
) -> AppConfig:
    path = Path(config_path) if config_path is not None else default_config_path()
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    if path.parent.name != "config":
        raise ConfigError(
            f"Config file must live in a directory named 'config': {path}"
        )
    root = path.parent.parent
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    data = _expand(loaded)
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    if day is not None:
        data["day"] = day
    paths = dict(_require_mapping(data, "paths"))
    if data_dir is not None:
        paths["data_dir"] = str(data_dir)
    if output_dir is not None:
        paths["output_dir"] = str(output_dir)
    if model_dir is not None:
        paths["model_dir"] = str(model_dir)
    data["paths"] = paths
    return _build(data, path, root)


def _build(data: dict[str, Any], path: Path, root: Path) -> AppConfig:
    classes = _classes(data.get("classes"))
    dataset = _require_mapping(data, "dataset")
    labels = _require_mapping(data, "labels")
    yolo = _require_mapping(data, "yolo")
    ncnn = _require_mapping(data, "ncnn")
    ae = _require_mapping(data, "autoencoder")
    infer = _require_mapping(data, "infer")
    paths = _require_mapping(data, "paths")
    label_format = str(labels.get("format", "")).strip()
    if label_format != LABEL_FORMAT:
        raise ConfigError(
            f"labels.format must be {LABEL_FORMAT!r}. Got {label_format!r}."
        )
    sources = _sources(infer.get("sources"))
    extensions = _extensions(dataset.get("image_extensions"))
    return AppConfig(
        config_path=path,
        repo_root=root,
        day=_as_int(data.get("day"), "day", allowed=frozenset({1, 2, 3})),
        seed=_as_int(data.get("seed"), "seed"),
        data_dir=_resolve(root, _nonempty(paths.get("data_dir"), "paths.data_dir")),
        output_dir=_resolve(root, _nonempty(paths.get("output_dir"), "paths.output_dir")),
        model_dir=_resolve(root, _nonempty(paths.get("model_dir"), "paths.model_dir")),
        classes=classes,
        class_to_id={name: index for index, name in enumerate(classes)},
        train_dirname=_nonempty(dataset.get("train_dirname"), "dataset.train_dirname"),
        test_dirname=_nonempty(dataset.get("test_dirname"), "dataset.test_dirname"),
        image_extensions=extensions,
        label_format=label_format,
        yolo_weights_name=_nonempty(yolo.get("weights_name"), "yolo.weights_name"),
        imgsz=_as_int(yolo.get("imgsz"), "yolo.imgsz", allowed=ALLOWED_IMGSZ),
        device=_nonempty(yolo.get("device"), "yolo.device"),
        epochs=_as_int(yolo.get("epochs"), "yolo.epochs", minimum=1),
        batch=_as_int(yolo.get("batch"), "yolo.batch", minimum=1),
        yolo_patience=_as_int(yolo.get("patience", 20), "yolo.patience", minimum=1),
        yolo_workers=_as_int(yolo.get("workers", 2), "yolo.workers", minimum=0),
        yolo_mosaic=_as_float(yolo.get("mosaic", 1.0), "yolo.mosaic", minimum=0.0, maximum=1.0),
        yolo_close_mosaic=_as_int(yolo.get("close_mosaic", 10), "yolo.close_mosaic", minimum=0),
        yolo_degrees=_as_float(yolo.get("degrees", 0.0), "yolo.degrees", minimum=0.0),
        yolo_scale=_as_float(yolo.get("scale", 0.5), "yolo.scale", minimum=0.0),
        yolo_flipud=_as_float(yolo.get("flipud", 0.0), "yolo.flipud", minimum=0.0, maximum=1.0),
        yolo_fliplr=_as_float(yolo.get("fliplr", 0.5), "yolo.fliplr", minimum=0.0, maximum=1.0),
        yolo_hsv_h=_as_float(yolo.get("hsv_h", 0.015), "yolo.hsv_h", minimum=0.0),
        yolo_hsv_s=_as_float(yolo.get("hsv_s", 0.7), "yolo.hsv_s", minimum=0.0),
        yolo_hsv_v=_as_float(yolo.get("hsv_v", 0.4), "yolo.hsv_v", minimum=0.0),
        yolo_copy_paste=_as_float(yolo.get("copy_paste", 0.0), "yolo.copy_paste", minimum=0.0, maximum=1.0),
        yolo_conf=_as_float(yolo.get("conf", 0.25), "yolo.conf", minimum=0.0, maximum=1.0),
        yolo_iou=_as_float(yolo.get("iou", 0.5), "yolo.iou", minimum=0.0, maximum=1.0),
        ncnn_threads=_as_int(ncnn.get("threads"), "ncnn.threads", minimum=1),
        ncnn_use_vulkan=_as_bool(ncnn.get("use_vulkan"), "ncnn.use_vulkan"),
        ae_enabled=_as_bool(ae.get("enabled"), "autoencoder.enabled"),
        ae_fusion_enabled=_as_bool(ae.get("fusion_enabled"), "autoencoder.fusion_enabled"),
        ae_crop_size=_as_int(ae.get("crop_size"), "autoencoder.crop_size", minimum=8),
        ae_latent_dim=_as_int(ae.get("latent_dim", 32), "autoencoder.latent_dim", minimum=4),
        ae_epochs=_as_int(ae.get("epochs", 40), "autoencoder.epochs", minimum=1),
        ae_batch=_as_int(ae.get("batch", 32), "autoencoder.batch", minimum=1),
        ae_lr=_as_float(ae.get("lr", 0.001), "autoencoder.lr", minimum=0.0),
        ae_val_fraction=_as_float(ae.get("val_fraction", 0.2), "autoencoder.val_fraction", minimum=0.0, maximum=0.5),
        ae_knn_k=_as_int(ae.get("knn_k", 5), "autoencoder.knn_k", minimum=1),
        ae_knn_share=_as_float(ae.get("knn_share", 0.6), "autoencoder.knn_share", minimum=0.0, maximum=1.0),
        ae_low_conf=_as_float(ae.get("low_conf", 0.45), "autoencoder.low_conf", minimum=0.0, maximum=1.0),
        ae_anomaly_quantile=_as_float(
            ae.get("anomaly_quantile", 0.95), "autoencoder.anomaly_quantile", minimum=0.0, maximum=1.0
        ),
        ae_confusable=_confusable(ae.get("confusable"), classes),
        infer_sources=sources,
        **_dataset_extra(dataset, classes),
        **_extract_config(data.get("extract")),
        **_synth_config(data.get("synth")),
    )


def _confusable(value: Any, classes: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    known = set(classes)
    if value is None:
        return tuple(pair for pair in _DEFAULT_CONFUSABLE if pair[0] in known and pair[1] in known)
    if not isinstance(value, list):
        raise ConfigError("autoencoder.confusable must be a list of class-name pairs.")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ConfigError("autoencoder.confusable entries must be two class names.")
        left, right = str(item[0]).strip(), str(item[1]).strip()
        if left not in known or right not in known:
            raise ConfigError(f"Confusable pair {(left, right)} is not in classes.")
        if left == right:
            raise ConfigError(f"Confusable pair {(left, right)} lists the same class twice.")
        pairs.append((left, right))
    return tuple(pairs)


def _dataset_extra(dataset: dict[str, Any], classes: tuple[str, ...]) -> dict[str, Any]:
    aliases = _aliases(dataset.get("sample_aliases") or {}, classes)
    return {
        "samples_dirname": str(dataset.get("samples_dirname") or "_sample"),
        "mock_dirname": str(dataset.get("mock_dirname") or "mock"),
        "day3_dirname": str(dataset.get("day3_dirname") or "day3"),
        "layout_suffix": str(dataset.get("layout_suffix") or ".layout.txt"),
        "sample_aliases": aliases,
    }


def _aliases(raw: Any, classes: tuple[str, ...]) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ConfigError("dataset.sample_aliases must be a mapping of filename stem to class name.")
    cleaned: dict[str, str] = {}
    for key, value in raw.items():
        name = str(value).strip()
        if name not in classes:
            known = ", ".join(classes)
            raise ConfigError(f"sample alias {key!r} points at unknown class {name!r}. Expected one of: {known}.")
        cleaned[_stem_key(str(key))] = name
    for name in classes:
        cleaned.setdefault(_stem_key(name), name)
    return cleaned


def _stem_key(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def _extract_config(raw: Any) -> dict[str, Any]:
    section = raw if isinstance(raw, dict) else {}
    backend = str(section.get("backend", "fastsam")).strip()
    if backend not in {"fastsam", "classical"}:
        raise ConfigError(f"extract.backend must be 'fastsam' or 'classical'. Got {backend!r}.")
    return {
        "extract_backend": backend,
        "extract_weights": str(section.get("weights_name", "FastSAM-s.pt")),
        "extract_imgsz": _as_int(section.get("imgsz", 640), "extract.imgsz", minimum=32),
        "extract_conf": _as_float(section.get("conf", 0.4), "extract.conf", minimum=0.0, maximum=1.0),
        "extract_min_area_frac": _as_float(section.get("min_area_frac", 0.001), "extract.min_area_frac", minimum=0.0),
        "extract_max_area_frac": _as_float(section.get("max_area_frac", 0.08), "extract.max_area_frac", minimum=0.0),
        "extract_max_aspect": _as_float(section.get("max_aspect", 2.6), "extract.max_aspect", minimum=1.0),
        "extract_lab_distance": _as_float(section.get("lab_distance", 28), "extract.lab_distance", minimum=0.0),
    }


def _synth_config(raw: Any) -> dict[str, Any]:
    section = raw if isinstance(raw, dict) else {}
    return {
        "synth_images_per_class": _as_int(section.get("images_per_class", 3), "synth.images_per_class", minimum=1),
        "synth_day3_per_pattern": _as_int(section.get("day3_images_per_pattern", 2), "synth.day3_images_per_pattern", minimum=1),
        "synth_pile_beans_min": _as_int(section.get("pile_beans_min", 16), "synth.pile_beans_min", minimum=1),
        "synth_pile_beans_max": _as_int(section.get("pile_beans_max", 28), "synth.pile_beans_max", minimum=1),
        "synth_mixed_classes_min": _as_int(section.get("mixed_classes_min", 3), "synth.mixed_classes_min", minimum=1),
        "synth_val_fraction": _as_float(section.get("val_fraction", 0.34), "synth.val_fraction", minimum=0.0, maximum=0.9),
        "synth_canvas_width": _as_int(section.get("canvas_width", 960), "synth.canvas_width", minimum=64),
        "synth_canvas_height": _as_int(section.get("canvas_height", 640), "synth.canvas_height", minimum=64),
    }


def _classes(value: Any) -> tuple[str, ...]:
    if list(value or []) != list(CLASS_NAMES):
        expected = ", ".join(CLASS_NAMES)
        raise ConfigError(
            "classes must be exactly "
            f"{expected} in that order (ids 0-6). Got {value!r}."
        )
    return CLASS_NAMES


def _sources(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("infer.sources must be a non-empty list of 'image' and/or 'folder'.")
    unknown = [item for item in value if item not in ALLOWED_SOURCES]
    if unknown:
        raise ConfigError(
            f"infer.sources has unsupported entries {unknown}. "
            "Allowed values are 'image' and 'folder'."
        )
    return tuple(str(item) for item in value)


def _extensions(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("dataset.image_extensions must be a non-empty list.")
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip().lower()
        if not text.startswith("."):
            text = f".{text}"
        cleaned.append(text)
    return tuple(cleaned)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(_env_sub, value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand(item) for key, item in value.items()}
    return value


def _env_sub(match: re.Match[str]) -> str:
    name = match.group(1)
    default = match.group(2)
    current = os.environ.get(name)
    if current is None or current.strip() == "":
        return "" if default is None else default
    return current


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"config missing key: {key}")
    return value


def _nonempty(value: Any, key: str) -> str:
    text = str(value).strip() if value is not None else ""
    if text == "":
        raise ConfigError(f"config missing key: {key}")
    return text


def _as_int(
    value: Any,
    key: str,
    *,
    allowed: frozenset[int] | None = None,
    minimum: int | None = None,
) -> int:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError, AttributeError) as exc:
        raise ConfigError(f"{key} must be an integer, got {value!r}.") from exc
    if allowed is not None and number not in allowed:
        choices = ", ".join(str(item) for item in sorted(allowed))
        raise ConfigError(f"{key} must be one of {choices}, got {number}.")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {number}.")
    return number


def _as_float(
    value: Any,
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError, AttributeError) as exc:
        raise ConfigError(f"{key} must be a number, got {value!r}.") from exc
    if minimum is not None and number < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {number}.")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{key} must be <= {maximum}, got {number}.")
    return number


def _as_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be true or false, got {value!r}.")
    return value
