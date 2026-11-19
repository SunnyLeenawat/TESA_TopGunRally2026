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
    ("black", "husk"),
    ("broken", "immature"),
)

_REGISTRY_FILE = Path(__file__).resolve().parents[2] / "config" / "classes.yaml"
ALLOWED_IMGSZ: frozenset[int] = frozenset({320, 480, 640})
ALLOWED_SOURCES: frozenset[str] = frozenset({"image", "folder"})
LABEL_FORMAT: str = "yolo_seg_txt"

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """The config file, an override, or a path is unusable."""


@dataclass(frozen=True)
class ClassRegistry:
    train_names: tuple[str, ...]
    submission_names: tuple[str, ...]
    aliases: dict[str, str]
    good_name: str


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
    submission_names: tuple[str, ...]
    class_aliases: dict[str, str]
    good_name: str
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
    mixed_dirname: str
    labels_filename: str
    separated_dirnames: tuple[str, ...]
    team: str
    http_base: str
    mqtt_host: str
    mqtt_port: int
    mqtt_topic: str
    mqtt_username: str
    mqtt_password: str
    http_images_path: str
    http_submit_path: str
    http_token: str
    http_mode: str
    http_timeout_s: float
    http_retries: int
    mqtt_timeout_s: float
    mqtt_retries: int
    mqtt_path: str
    mqtt_track: str
    mqtt_event: str
    mqtt_client_id: str
    day3_fold_count: int
    day3_val_fraction: float
    day3_preview_max_side: int
    day3_baseline_epochs: int
    day3_mosaic: float
    day3_degrees: float
    day3_scale: float
    day3_roi_margin_frac: float
    day3_roi_dark_delta: float
    day3_roi_fallback_frac: float
    day3_infer_conf: float
    day3_dedup_iou: float
    day3_min_area: int
    day3_refine: bool
    day3_s5_epochs: int
    day3_pile_count: int
    day3_pile_beans_min: int
    day3_pile_beans_max: int
    day3_pile_canvas: int

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


def load_class_registry(path: Path) -> ClassRegistry:
    """Read config/classes.yaml. Train ids are the index of train_order."""
    if not path.is_file():
        raise ConfigError(f"Class registry not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"Class registry root must be a mapping: {path}")
    train = _name_list(loaded.get("train_order"), "train_order")
    submission = _name_list(loaded.get("submission_order"), "submission_order")
    if set(train) != set(submission):
        raise ConfigError("train_order and submission_order must contain the same class names.")
    good = str(loaded.get("good_name", "good")).strip()
    if good not in train:
        raise ConfigError(f"good_name {good!r} is not in train_order.")
    aliases_raw = loaded.get("aliases") or {}
    if not isinstance(aliases_raw, dict):
        raise ConfigError("aliases must be a mapping of old name to train name.")
    aliases: dict[str, str] = {}
    for key, value in aliases_raw.items():
        source = str(key).strip()
        target = str(value).strip()
        if target not in train:
            raise ConfigError(f"Alias {source!r} points at unknown class {target!r}.")
        if source == target:
            raise ConfigError(f"Alias {source!r} points at itself.")
        aliases[source] = target
    return ClassRegistry(train, submission, aliases, good)


def _name_list(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{key} must be a non-empty list of class names.")
    names = tuple(str(item).strip() for item in value)
    if any(name == "" for name in names):
        raise ConfigError(f"{key} contains an empty class name.")
    if len(set(names)) != len(names):
        raise ConfigError(f"{key} contains a duplicate class name.")
    return names


def _import_train_names() -> tuple[str, ...]:
    if not _REGISTRY_FILE.is_file():
        return ()
    return load_class_registry(_REGISTRY_FILE).train_names


CLASS_NAMES: tuple[str, ...] = _import_train_names()


def _build(data: dict[str, Any], path: Path, root: Path) -> AppConfig:
    registry = _registry(data.get("registry"), path)
    classes = _classes(data.get("classes"), None if registry is None else registry.train_names)
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
        submission_names=classes if registry is None else registry.submission_names,
        class_aliases={} if registry is None else dict(registry.aliases),
        good_name=("good" if "good" in classes else classes[0]) if registry is None else registry.good_name,
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
        **_day3(data.get("day3")),
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


def _registry(value: Any, config_path: Path) -> ClassRegistry | None:
    if value is None or str(value).strip() == "":
        return None
    registry_path = Path(str(value).strip())
    if not registry_path.is_absolute():
        registry_path = config_path.parent / registry_path
    return load_class_registry(registry_path)


def _classes(value: Any, registry_names: tuple[str, ...] | None) -> tuple[str, ...]:
    if registry_names is not None:
        if value is not None:
            given = _name_list(value, "classes")
            if given != registry_names:
                expected = ", ".join(registry_names)
                raise ConfigError(
                    f"classes must match the registry train_order ({expected}). Got {list(given)!r}."
                )
        return registry_names
    return _name_list(value, "classes")


def _day3(raw: Any) -> dict[str, Any]:
    section = raw if isinstance(raw, dict) else {}
    separated = section.get("separated_dirnames", ["dataset_TESA", "train"])
    if not isinstance(separated, list) or not separated:
        raise ConfigError("day3.separated_dirnames must be a non-empty list.")
    names = tuple(str(item).strip() for item in separated)
    if any(name == "" for name in names):
        raise ConfigError("day3.separated_dirnames contains an empty name.")
    team = str(section.get("team") or "team_55").strip()
    if team == "":
        raise ConfigError("day3.team must not be empty.")
    return {
        "mixed_dirname": str(section.get("mixed_dirname") or "mix").strip(),
        "labels_filename": str(section.get("labels_filename") or "labels.json").strip(),
        "separated_dirnames": names,
        "team": team,
        "http_base": str(section.get("http_base") or "").strip(),
        "mqtt_host": str(section.get("mqtt_host") or "127.0.0.1").strip(),
        "mqtt_port": _as_int(section.get("mqtt_port", 1883), "day3.mqtt_port", minimum=1),
        "mqtt_topic": str(section.get("mqtt_topic") or "").strip(),
        "mqtt_username": str(section.get("mqtt_username") or "").strip(),
        "mqtt_password": str(section.get("mqtt_password") or ""),
        "day3_fold_count": _as_int(section.get("fold_count", 4), "day3.fold_count", minimum=2),
        "day3_val_fraction": _as_float(
            section.get("val_fraction", 0.25), "day3.val_fraction", minimum=0.0, maximum=0.9
        ),
        "day3_preview_max_side": _as_int(
            section.get("preview_max_side", 960), "day3.preview_max_side", minimum=64
        ),
        "day3_baseline_epochs": _as_int(
            section.get("baseline_epochs", 30), "day3.baseline_epochs", minimum=1
        ),
        "day3_mosaic": _as_float(section.get("mosaic", 0.0), "day3.mosaic", minimum=0.0, maximum=1.0),
        "day3_degrees": _as_float(section.get("degrees", 180.0), "day3.degrees", minimum=0.0),
        "day3_scale": _as_float(section.get("scale", 0.15), "day3.scale", minimum=0.0),
        "day3_roi_margin_frac": _as_float(
            section.get("roi_margin_frac", 0.15), "day3.roi_margin_frac", minimum=0.0, maximum=0.5
        ),
        "day3_roi_dark_delta": _as_float(
            section.get("roi_dark_delta", 12.0), "day3.roi_dark_delta", minimum=1.0
        ),
        "day3_roi_fallback_frac": _as_float(
            section.get("roi_fallback_frac", 0.7), "day3.roi_fallback_frac", minimum=0.2, maximum=1.0
        ),
        "day3_infer_conf": _as_float(
            section.get("infer_conf", 0.02), "day3.infer_conf", minimum=0.0, maximum=1.0
        ),
        "day3_dedup_iou": _as_float(
            section.get("dedup_iou", 0.5), "day3.dedup_iou", minimum=0.0, maximum=1.0
        ),
        "day3_min_area": _as_int(section.get("min_area", 64), "day3.min_area", minimum=1),
        "day3_refine": _as_bool(section.get("refine", False), "day3.refine"),
        "day3_s5_epochs": _as_int(section.get("s5_epochs", 40), "day3.s5_epochs", minimum=1),
        "day3_pile_count": _as_int(section.get("pile_count", 40), "day3.pile_count", minimum=1),
        "day3_pile_beans_min": _as_int(section.get("pile_beans_min", 8), "day3.pile_beans_min", minimum=1),
        "day3_pile_beans_max": _as_int(section.get("pile_beans_max", 36), "day3.pile_beans_max", minimum=1),
        "day3_pile_canvas": _as_int(section.get("pile_canvas", 1280), "day3.pile_canvas", minimum=256),
        "http_images_path": str(section.get("http_images_path") or "/images").strip(),
        "http_submit_path": str(section.get("http_submit_path") or "/submit").strip(),
        "http_token": str(section.get("http_token") or ""),
        "http_timeout_s": _as_float(section.get("http_timeout_s", 60), "day3.http_timeout_s", minimum=1.0),
        "http_retries": _as_int(section.get("http_retries", 3), "day3.http_retries", minimum=1),
        "mqtt_timeout_s": _as_float(section.get("mqtt_timeout_s", 10), "day3.mqtt_timeout_s", minimum=1.0),
        "mqtt_retries": _as_int(section.get("mqtt_retries", 3), "day3.mqtt_retries", minimum=1),
        "http_mode": str(section.get("http_mode") or "train").strip(),
        "mqtt_path": str(section.get("mqtt_path") or "/mqtt/").strip(),
        "mqtt_track": str(section.get("mqtt_track") or "EdgeAI").strip(),
        "mqtt_event": str(section.get("mqtt_event") or "system-testing").strip(),
        "mqtt_client_id": str(section.get("mqtt_client_id") or "TGR-55:device-1").strip(),
    }


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
