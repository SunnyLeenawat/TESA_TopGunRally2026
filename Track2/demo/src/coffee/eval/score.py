"""Local Day 3 judge: IoU, counting, latency, and the MQTT file check.

IoU uses one-to-one matching. Unmatched ground truth and extra predictions
count as IoU 0. Counting ignores the IoU value: a pair matches with no spatial
threshold. Class-agnostic and class-aware results are both reported.
P for the latency penalty is IoU points plus counting points (max 30).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from coffee.config import AppConfig
from coffee.data.day3_labels import load_string_labels
from coffee.eval.validate import validate_outputs


@dataclass(frozen=True)
class ScoreReport:
    payload: dict[str, object]

    @property
    def projected_agnostic(self) -> float:
        return float(self.payload["projected_agnostic"])


def score_folder(cfg: AppConfig, pred_dir: Path, labels_path: Path, image_dir: Path) -> ScoreReport:
    images = _images(image_dir, cfg.image_extensions)
    validate_outputs(pred_dir, images, cfg.submission_names, cfg.good_name)
    ground_truth = {item.image_name: item for item in load_string_labels(labels_path, cfg)}
    predictions = {
        entry["image_name"]: entry
        for entry in json.loads((pred_dir / "predictions.json").read_text(encoding="utf-8"))
    }
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    mqtt_ok = 0
    for path in images:
        label = ground_truth.get(path.name)
        if label is None:
            raise FileNotFoundError(f"{labels_path} has no entry for {path.name}.")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read {path}.")
        pred_mask = cv2.imread(str(pred_dir / "seg_mask" / f"{path.stem}.png"), cv2.IMREAD_UNCHANGED)
        pred_objects = predictions[path.name]["objects"]
        gt_masks, gt_names = _gt_masks(label, image.shape[:2])
        pred_masks, pred_names = _pred_masks(pred_mask, pred_objects)
        row = _image_row(path.name, gt_masks, gt_names, pred_masks, pred_names)
        message = json.loads((pred_dir / "mqtt" / f"{path.stem}.json").read_text(encoding="utf-8"))
        latencies.append(int(message["latency_ms"]) / 1000.0)
        mask_counts = _names_to_counts(pred_names, cfg.submission_names)
        if message.get("counts") == mask_counts:
            mqtt_ok += 1
            row["mqtt_ok"] = True
        else:
            row["mqtt_ok"] = False
        rows.append(row)
    median_s = float(np.median(latencies)) if latencies else 0.0
    agnostic_iou = _mean(rows, "iou_hungarian_agnostic")
    aware_iou = _mean(rows, "iou_hungarian_aware")
    greedy_iou = _mean(rows, "iou_greedy_agnostic")
    greedy_aware = _mean(rows, "iou_greedy_aware")
    agnostic_count = _mean(rows, "count_agnostic")
    aware_count = _mean(rows, "count_aware")
    p1_agnostic = 15.0 * agnostic_iou
    p2_agnostic = 15.0 * agnostic_count
    p1_aware = 15.0 * aware_iou
    p2_aware = 15.0 * aware_count
    mqtt = 5.0 * (mqtt_ok / len(images) if images else 1.0)
    latency_agnostic = latency_points(median_s, p1_agnostic + p2_agnostic)
    latency_aware = latency_points(median_s, p1_aware + p2_aware)
    payload = {
        "images": rows,
        "median_s": median_s,
        "iou_hungarian_agnostic": agnostic_iou,
        "iou_hungarian_aware": aware_iou,
        "iou_greedy_agnostic": greedy_iou,
        "iou_greedy_aware": greedy_aware,
        "count_agnostic": agnostic_count,
        "count_aware": aware_count,
        "p1_agnostic": p1_agnostic,
        "p2_agnostic": p2_agnostic,
        "p1_aware": p1_aware,
        "p2_aware": p2_aware,
        "latency_agnostic": latency_agnostic,
        "latency_aware": latency_aware,
        "mqtt": mqtt,
        "projected_agnostic": p1_agnostic + p2_agnostic + latency_agnostic + mqtt,
        "projected_aware": p1_aware + p2_aware + latency_aware + mqtt,
        "leakage_warning": (
            "These images share beans with training shots from other angles. "
            "The score is optimistic."
        ),
    }
    return ScoreReport(payload)


def latency_points(median_s: float, quality_points: float) -> float:
    """25 points at 1.00 s or faster. Each extra ceil(t - 1) second costs 5. Quality can scale it down."""
    overrun = max(0, math.ceil(median_s - 1.0 - 1e-12))
    points = max(0.0, 25.0 - 5.0 * overrun)
    half = 15.0
    if quality_points < half:
        points *= quality_points / half
    return points


def mean_iou(matched: list[float], n_gt: int, n_pred: int) -> float:
    if n_gt == 0 and n_pred == 0:
        return 1.0
    denominator = n_gt + n_pred - len(matched)
    if denominator <= 0:
        return 1.0
    return float(sum(matched) / denominator)


def count_ratio(matched: int, n_gt: int, n_pred: int) -> float:
    if n_gt == 0 and n_pred == 0:
        return 1.0
    return matched / max(n_gt, n_pred)


def greedy_pairs(score: np.ndarray) -> list[tuple[int, int]]:
    if score.size == 0:
        return []
    order = np.dstack(np.unravel_index(np.argsort(score, axis=None)[::-1], score.shape))[0]
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for row, col in order:
        row_i, col_i = int(row), int(col)
        if row_i in used_rows or col_i in used_cols:
            continue
        if score[row_i, col_i] <= 0:
            break
        used_rows.add(row_i)
        used_cols.add(col_i)
        pairs.append((row_i, col_i))
    return pairs


def hungarian_pairs(score: np.ndarray) -> list[tuple[int, int]]:
    if score.size == 0:
        return []
    cost = float(score.max()) - score
    rows, cols = _linear_sum_assignment(cost)
    return [(row, col) for row, col in zip(rows, cols) if score[row, col] > 0]


def _image_row(
    name: str,
    gt_masks: list[np.ndarray],
    gt_names: list[str],
    pred_masks: list[np.ndarray],
    pred_names: list[str],
) -> dict[str, object]:
    iou = _iou_matrix(gt_masks, pred_masks)
    aware = iou.copy()
    for row, gt_name in enumerate(gt_names):
        for col, pred_name in enumerate(pred_names):
            if gt_name != pred_name:
                aware[row, col] = 0.0
    hungarian = hungarian_pairs(iou)
    hungarian_aware = hungarian_pairs(aware)
    greedy = greedy_pairs(iou)
    greedy_aware = greedy_pairs(aware)
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)
    agnostic_matches = min(n_gt, n_pred)
    aware_matches = _class_matches(gt_names, pred_names)
    return {
        "image": name,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "iou_hungarian_agnostic": mean_iou([float(iou[r, c]) for r, c in hungarian], n_gt, n_pred),
        "iou_hungarian_aware": mean_iou([float(aware[r, c]) for r, c in hungarian_aware], n_gt, n_pred),
        "iou_greedy_agnostic": mean_iou([float(iou[r, c]) for r, c in greedy], n_gt, n_pred),
        "iou_greedy_aware": mean_iou([float(aware[r, c]) for r, c in greedy_aware], n_gt, n_pred),
        "count_agnostic": count_ratio(agnostic_matches, n_gt, n_pred),
        "count_aware": count_ratio(aware_matches, n_gt, n_pred),
    }


def _class_matches(gt_names: list[str], pred_names: list[str]) -> int:
    gt_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    for name in gt_names:
        gt_counts[name] = gt_counts.get(name, 0) + 1
    for name in pred_names:
        pred_counts[name] = pred_counts.get(name, 0) + 1
    return sum(min(gt_counts.get(name, 0), pred_counts.get(name, 0)) for name in set(gt_counts) | set(pred_counts))


def _iou_matrix(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> np.ndarray:
    matrix = np.zeros((len(gt_masks), len(pred_masks)), np.float64)
    for row, gt_mask in enumerate(gt_masks):
        gt_area = int(np.count_nonzero(gt_mask))
        for col, pred_mask in enumerate(pred_masks):
            intersection = int(np.count_nonzero(gt_mask & pred_mask))
            if intersection == 0:
                continue
            union = gt_area + int(np.count_nonzero(pred_mask)) - intersection
            matrix[row, col] = intersection / union if union else 0.0
    return matrix


def _gt_masks(label, shape: tuple[int, int]) -> tuple[list[np.ndarray], list[str]]:
    masks: list[np.ndarray] = []
    names: list[str] = []
    for bean in label.beans:
        mask = np.zeros(shape, np.uint8)
        contour = np.round(bean.polygon_xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [contour], 1)
        if int(mask.sum()) == 0:
            continue
        masks.append(mask.astype(bool))
        names.append(bean.class_name)
    return masks, names


def _pred_masks(mask: np.ndarray, objects: list[dict]) -> tuple[list[np.ndarray], list[str]]:
    masks: list[np.ndarray] = []
    names: list[str] = []
    for obj in objects:
        masks.append(mask == int(obj["id"]))
        names.append(str(obj["class"]))
    return masks, names


def _names_to_counts(names: list[str], class_names: tuple[str, ...]) -> dict[str, int]:
    counts = {name: 0 for name in class_names}
    for name in names:
        counts[name] += 1
    return counts


def _images(folder: Path, extensions: tuple[str, ...]) -> list[Path]:
    if folder.is_file():
        return [folder]
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def _mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 1.0
    return float(np.mean([float(row[key]) for row in rows]))


def _linear_sum_assignment(cost: np.ndarray) -> tuple[list[int], list[int]]:
    """Minimum-cost assignment. Rectangular inputs are padded."""
    matrix = np.asarray(cost, dtype=np.float64)
    height, width = matrix.shape
    transposed = height > width
    if transposed:
        matrix = matrix.T
        height, width = matrix.shape
    potential_u = np.zeros(height + 1, dtype=np.float64)
    potential_v = np.zeros(width + 1, dtype=np.float64)
    assignment = np.zeros(width + 1, dtype=np.int64)
    way = np.zeros(width + 1, dtype=np.int64)
    for row in range(1, height + 1):
        assignment[0] = row
        column0 = 0
        minimum = np.full(width + 1, np.inf)
        used = np.zeros(width + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = int(assignment[column0])
            delta = np.inf
            column1 = 0
            for column in range(1, width + 1):
                if used[column]:
                    continue
                current = matrix[row0 - 1, column - 1] - potential_u[row0] - potential_v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = float(minimum[column])
                    column1 = column
            for column in range(width + 1):
                if used[column]:
                    potential_u[assignment[column]] += delta
                    potential_v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if assignment[column0] == 0:
                break
        while True:
            column1 = int(way[column0])
            assignment[column0] = assignment[column1]
            column0 = column1
            if column0 == 0:
                break
    rows: list[int] = []
    cols: list[int] = []
    for column in range(1, width + 1):
        if assignment[column] == 0:
            continue
        rows.append(int(assignment[column] - 1))
        cols.append(column - 1)
    if transposed:
        return cols, rows
    return rows, cols
