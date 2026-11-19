"""Check a submitted tree: predictions, seg_mask ids, and mqtt counts must agree."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SUBMISSION_NAMES = ("good", "shell", "black", "husk", "broken", "parchment", "immature")


def check_tree(root: Path) -> list[str]:
    """Return problems. An empty list means the invariant holds."""
    prediction_files = list(root.rglob("predictions.json"))
    if len(prediction_files) != 1:
        return [f"Expected one predictions.json, found {len(prediction_files)}."]
    try:
        predictions = json.loads(prediction_files[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"predictions.json is not JSON: {exc}"]
    if not isinstance(predictions, list):
        return ["predictions.json must be a list."]
    problems: list[str] = []
    for entry in predictions:
        if not isinstance(entry, dict):
            problems.append("A predictions entry is not an object.")
            continue
        name = str(entry.get("image_name", ""))
        objects = entry.get("objects")
        if not name or not isinstance(objects, list):
            problems.append(f"Bad predictions entry for {name!r}.")
            continue
        stem = Path(name).stem
        counts = {label: 0 for label in SUBMISSION_NAMES}
        ids: list[int] = []
        for obj in objects:
            if not isinstance(obj, dict):
                problems.append(f"{name} has a non-object bean.")
                continue
            if "score" in obj or "confidence" in obj:
                problems.append(f"{name} object includes a confidence field.")
            class_name = obj.get("class")
            if class_name not in counts:
                problems.append(f"{name} class {class_name!r} is not a submission name.")
            else:
                counts[str(class_name)] += 1
            if not isinstance(obj.get("id"), int):
                problems.append(f"{name} id must be an integer.")
            else:
                ids.append(int(obj["id"]))
        if ids != list(range(1, len(ids) + 1)):
            problems.append(f"{name} ids must be 1..N, got {ids}.")
        problems.extend(_mask_ids(root, stem, ids))
        problems.extend(_mqtt_counts(root, stem, name, counts))
    return problems


def _mask_ids(root: Path, stem: str, ids: list[int]) -> list[str]:
    matches = [path for path in root.rglob(f"{stem}.png") if path.parent.name == "seg_mask"]
    if len(matches) != 1:
        return [f"Expected one seg_mask for {stem}, found {len(matches)}."]
    mask = cv2.imread(str(matches[0]), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        return [f"seg_mask {stem} is not a single-channel image."]
    found = [int(value) for value in np.unique(mask) if int(value) != 0]
    if found != ids:
        return [f"{stem} seg_mask ids {found} != prediction ids {ids}."]
    return []


def _mqtt_counts(root: Path, stem: str, image_name: str, counts: dict[str, int]) -> list[str]:
    matches = [path for path in root.rglob(f"{stem}.json") if path.parent.name == "mqtt"]
    if len(matches) != 1:
        return [f"Expected one mqtt file for {stem}, found {len(matches)}."]
    try:
        message = json.loads(matches[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{stem} mqtt JSON is invalid: {exc}"]
    if message.get("image") != image_name:
        return [f"{stem} mqtt image is {message.get('image')!r}."]
    got = message.get("counts")
    if got != counts:
        return [f"{stem} mqtt counts {got} != predictions {counts}."]
    return []
