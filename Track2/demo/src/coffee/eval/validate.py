"""Check the Day 3 output contract. Counts, mask ids, and polygons must agree."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from coffee.infer.instances import defect_ratio

_OBJECT_KEYS = {"id", "class", "polygon"}


class ContractError(ValueError):
    """The written outputs disagree with each other or with the input images."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(f"{len(problems)} contract error(s). First: {problems[0]}")


def validate_outputs(
    out_dir: Path,
    images: list[Path],
    class_names: tuple[str, ...],
    good_name: str,
) -> None:
    """Raise ContractError when the folder breaks the output contract."""
    problems = _problems(out_dir, images, class_names, good_name)
    if problems:
        raise ContractError(problems)


def _problems(
    out_dir: Path,
    images: list[Path],
    class_names: tuple[str, ...],
    good_name: str,
) -> list[str]:
    problems: list[str] = []
    prediction_path = out_dir / "predictions.json"
    if not prediction_path.is_file():
        return [f"Missing {prediction_path}."]
    try:
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"predictions.json is not valid JSON: {exc}"]
    if not isinstance(predictions, list):
        return ["predictions.json must be a list."]
    by_name = {}
    for entry in predictions:
        if not isinstance(entry, dict) or "image_name" not in entry:
            problems.append("A predictions.json entry has no image_name.")
            continue
        by_name[str(entry["image_name"])] = entry
    expected = [path.name for path in images]
    if sorted(by_name) != sorted(expected):
        problems.append(
            f"predictions.json images {sorted(by_name)} do not match the input {sorted(expected)}."
        )
    for path in images:
        problems.extend(_one_image(out_dir, path, by_name.get(path.name), class_names, good_name))
    return problems


def _one_image(
    out_dir: Path,
    image_path: Path,
    entry: dict | None,
    class_names: tuple[str, ...],
    good_name: str,
) -> list[str]:
    problems: list[str] = []
    stem = image_path.stem
    if entry is None:
        return [f"{image_path.name} is missing from predictions.json."]
    objects = entry.get("objects")
    if not isinstance(objects, list):
        return [f"{image_path.name} objects must be a list."]
    mask_path = out_dir / "seg_mask" / f"{stem}.png"
    mqtt_path = out_dir / "mqtt" / f"{stem}.json"
    vis_path = out_dir / "vis" / f"{stem}.png"
    for required in (mask_path, mqtt_path, vis_path):
        if not required.is_file():
            problems.append(f"Missing {required}.")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) if mask_path.is_file() else None
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        problems.append(f"Could not read input image {image_path}.")
        return problems
    if mask is None:
        return problems
    if mask.shape[:2] != image.shape[:2]:
        problems.append(f"{stem} seg_mask shape {mask.shape[:2]} != image {image.shape[:2]}.")
    if mask.ndim != 2:
        problems.append(f"{stem} seg_mask must be a single channel.")
        return problems
    ids = [int(value) for value in np.unique(mask) if int(value) != 0]
    object_ids: list[int] = []
    object_counts = {name: 0 for name in class_names}
    for obj in objects:
        if not isinstance(obj, dict):
            problems.append(f"{stem} has an object that is not a mapping.")
            continue
        extra = set(obj) - _OBJECT_KEYS
        if extra:
            problems.append(f"{stem} object has extra fields {sorted(extra)}.")
        if "score" in obj or "confidence" in obj:
            problems.append(f"{stem} object includes a confidence field.")
        class_name = obj.get("class")
        if class_name not in object_counts:
            problems.append(f"{stem} class {class_name!r} is not a submission name.")
        else:
            object_counts[str(class_name)] += 1
        if not isinstance(obj.get("id"), int):
            problems.append(f"{stem} object id must be an integer.")
            continue
        object_ids.append(int(obj["id"]))
        polygon = obj.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            problems.append(f"{stem} id {obj.get('id')} polygon needs at least 3 points.")
    if object_ids != list(range(1, len(object_ids) + 1)):
        problems.append(f"{stem} object ids must be 1..N in order, got {object_ids}.")
    if ids != object_ids:
        problems.append(f"{stem} seg_mask ids {ids} != prediction ids {object_ids}.")
    mask_counts = {name: 0 for name in class_names}
    id_to_class = {int(obj["id"]): str(obj["class"]) for obj in objects if isinstance(obj, dict) and isinstance(obj.get("id"), int)}
    for value in ids:
        class_name = id_to_class.get(value)
        if class_name in mask_counts:
            mask_counts[class_name] += 1
    if not mqtt_path.is_file():
        return problems
    try:
        message = json.loads(mqtt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"{stem} mqtt JSON is invalid: {exc}")
        return problems
    if message.get("image") != image_path.name:
        problems.append(f"{stem} mqtt image is {message.get('image')!r}.")
    counts = message.get("counts")
    if not isinstance(counts, dict) or list(counts) != list(class_names):
        problems.append(f"{stem} counts keys must be {list(class_names)}.")
        return problems
    if counts != object_counts or counts != mask_counts:
        problems.append(
            f"{stem} counts disagree. mqtt={counts} predictions={object_counts} mask={mask_counts}."
        )
    expected_ratio = defect_ratio({str(key): int(value) for key, value in counts.items()}, good_name)
    actual = message.get("defect_ratio")
    if not isinstance(actual, (int, float)) or abs(float(actual) - expected_ratio) > 1e-6:
        problems.append(f"{stem} defect_ratio {actual} != {expected_ratio}.")
    if not isinstance(message.get("latency_ms"), int) or int(message["latency_ms"]) < 0:
        problems.append(f"{stem} latency_ms must be a non-negative integer.")
    return problems
