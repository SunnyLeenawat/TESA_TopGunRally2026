"""Paste masked bean crops onto a wooden canvas and emit YOLO-seg labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from coffee.config import AppConfig
from coffee.data.beans import Instance
from coffee.data.labels import write_layout, write_yolo_seg


@dataclass
class Placed:
    class_name: str
    polygon_xy: np.ndarray


def wood_canvas(wood_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(wood_bgr, (width, height), interpolation=cv2.INTER_AREA)


def wood_patch(image: np.ndarray, instances: list[Instance]) -> np.ndarray:
    """Crop the largest bean-free patch of table. Inpainting a full photo smears the grain."""
    height, width = image.shape[:2]
    occupied = np.zeros((height, width), np.uint8)
    for instance in instances:
        points = np.round(instance.polygon_xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(occupied, [points], 255)
    border = max(4, int(round(0.04 * min(height, width))))
    occupied[:border] = 255
    occupied[-border:] = 255
    occupied[:, :border] = 255
    occupied[:, -border:] = 255
    occupied = cv2.dilate(occupied, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    for side in (64, 48, 40):
        best: np.ndarray | None = None
        best_score = -1.0
        step = max(8, side // 3)
        for y in range(0, height - side + 1, step):
            for x in range(0, width - side + 1, step):
                if occupied[y : y + side, x : x + side].any():
                    continue
                patch = image[y : y + side, x : x + side]
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                # Bean tips are bright bumps. Wood grain is striped but even along a row.
                bright = float((gray > float(np.median(gray)) + 22).mean())
                if bright > 0.15:
                    continue
                score = float(np.std(gray))
                if score > best_score:
                    best_score = score
                    best = patch.copy()
        if best is not None:
            return best
    raise RuntimeError("No empty wood patch found. The photo may be covered edge to edge.")


def tile_wood(patch: np.ndarray, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    canvas = np.zeros((height, width, 3), np.uint8)
    patch_h, patch_w = patch.shape[:2]
    y = 0
    while y < height:
        x = 0
        row_flip = bool(rng.integers(0, 2))
        while x < width:
            tile = patch.copy()
            gain = float(rng.uniform(0.9, 1.1))
            tile = np.clip(tile.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            if row_flip ^ bool(rng.integers(0, 2)):
                tile = cv2.flip(tile, 1)
            y1 = min(height, y + patch_h)
            x1 = min(width, x + patch_w)
            canvas[y:y1, x:x1] = tile[: y1 - y, : x1 - x]
            x += patch_w
        y += patch_h
    return canvas


def render_neat(
    background: np.ndarray,
    crops: list[Instance],
    class_name: str,
    pattern: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[Placed]]:
    slots = _slots(pattern, background.shape[1], background.shape[0])
    return _render(background, crops, [class_name] * len(slots), slots, rng, neat=True)


def pattern_slots(pattern: str, width: int, height: int) -> list[tuple[int, int]]:
    return _slots(pattern, width, height)


def render_assigned(
    background: np.ndarray,
    crops_by_class: dict[str, list[Instance]],
    class_names: list[str],
    pattern: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[Placed]]:
    slots = _slots(pattern, background.shape[1], background.shape[0])
    if len(class_names) != len(slots):
        raise RuntimeError(
            f"Pattern {pattern} has {len(slots)} slots but {len(class_names)} class names were given."
        )
    selected = [
        crops_by_class[name][int(rng.integers(0, len(crops_by_class[name])))]
        for name in class_names
    ]
    return _render_selected(background, selected, class_names, slots, rng, neat=True)


def render_pile(
    background: np.ndarray,
    crops_by_class: dict[str, list[Instance]],
    class_names: list[str],
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[Placed]]:
    height, width = background.shape[:2]
    margin_x = int(0.08 * width)
    margin_y = int(0.08 * height)
    slots = [
        (
            int(rng.integers(margin_x, width - margin_x)),
            int(rng.integers(margin_y, height - margin_y)),
        )
        for _ in range(count)
    ]
    chosen = [str(rng.choice(class_names)) for _ in slots]
    pool = [crops_by_class[name] for name in chosen]
    flat = [pool[index][int(rng.integers(0, len(pool[index])))] for index in range(len(slots))]
    return _render_selected(background, flat, chosen, slots, rng, neat=False)


def draw_placed(image: np.ndarray, placed: list[Placed]) -> np.ndarray:
    canvas = image.copy()
    for item in placed:
        points = np.round(item.polygon_xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], isClosed=True, color=(0, 0, 255), thickness=2)
    return canvas


def save_scene(
    image: np.ndarray,
    placed: list[Placed],
    image_path: Path,
    cfg: AppConfig,
    *,
    layout: bool,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"Could not write image: {image_path}")
    class_ids = [cfg.class_to_id[item.class_name] for item in placed]
    polygons = [item.polygon_xy for item in placed]
    write_yolo_seg(
        image_path.with_suffix(".txt"),
        class_ids,
        polygons,
        image.shape[1],
        image.shape[0],
    )
    if layout:
        write_layout(
            image_path.with_name(image_path.stem + cfg.layout_suffix),
            [item.class_name for item in placed],
        )


def _render(
    background: np.ndarray,
    crops: list[Instance],
    class_names: list[str],
    slots: list[tuple[int, int]],
    rng: np.random.Generator,
    *,
    neat: bool,
) -> tuple[np.ndarray, list[Placed]]:
    if not crops:
        raise RuntimeError("Cannot paste a scene with no bean crops.")
    selected = [crops[int(rng.integers(0, len(crops)))] for _ in slots]
    return _render_selected(background, selected, class_names, slots, rng, neat=neat)


def _render_selected(
    background: np.ndarray,
    crops: list[Instance],
    class_names: list[str],
    slots: list[tuple[int, int]],
    rng: np.random.Generator,
    *,
    neat: bool,
) -> tuple[np.ndarray, list[Placed]]:
    canvas = background.copy()
    height, width = canvas.shape[:2]
    layers: list[tuple[str, np.ndarray]] = []
    for crop, class_name, (cx, cy) in zip(crops, class_names, slots):
        angle = float(rng.uniform(-12, 12) if neat else rng.uniform(0, 360))
        scale = float(rng.uniform(0.9, 1.05) if neat else rng.uniform(0.75, 1.15))
        if neat:
            scale *= _fit_scale(crop, width, height, len(slots))
        rotated_bgr, rotated_mask = _rotate(crop.crop_bgr, crop.crop_mask, angle, scale)
        rotated_bgr = _jitter(rotated_bgr, rotated_mask, rng)
        stamped = _stamp(canvas, rotated_bgr, rotated_mask, cx, cy, shadow=not neat)
        if stamped is None:
            continue
        layers.append((class_name, stamped))
    placed = _visible_polygons(layers, width, height)
    return canvas, placed


def _visible_polygons(
    layers: list[tuple[str, np.ndarray]],
    width: int,
    height: int,
) -> list[Placed]:
    """Later pastes cover earlier ones. Labels follow the visible pixels."""
    if not layers:
        return []
    occupied = np.zeros((height, width), np.uint8)
    visible: list[tuple[str, np.ndarray]] = []
    for class_name, mask in reversed(layers):
        shown = cv2.bitwise_and(mask, cv2.bitwise_not(occupied))
        occupied = cv2.bitwise_or(occupied, mask)
        original = int(mask.sum() // 255)
        remain = int(shown.sum() // 255)
        if original == 0 or remain < 0.35 * original:
            continue
        visible.append((class_name, shown))
    placed: list[Placed] = []
    for class_name, mask in reversed(visible):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 20:
            continue
        polygon = contour.reshape(-1, 2).astype(np.float32)
        placed.append(Placed(class_name=class_name, polygon_xy=polygon))
    return placed


def _stamp(
    canvas: np.ndarray,
    crop: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
    *,
    shadow: bool,
) -> np.ndarray | None:
    crop_h, crop_w = mask.shape[:2]
    height, width = canvas.shape[:2]
    x0 = int(cx - crop_w / 2)
    y0 = int(cy - crop_h / 2)
    x1, y1 = x0 + crop_w, y0 + crop_h
    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    src_x1, src_y1 = crop_w - max(0, x1 - width), crop_h - max(0, y1 - height)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(width, x1), min(height, y1)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return None
    patch_mask = mask[src_y0:src_y1, src_x0:src_x1]
    if int(patch_mask.sum()) < 20 * 255:
        return None
    if shadow:
        _shadow(canvas, patch_mask, dst_x0 + 4, dst_y0 + 6)
    patch = crop[src_y0:src_y1, src_x0:src_x1]
    alpha = (patch_mask.astype(np.float32) / 255.0)[:, :, None]
    region = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(alpha * patch + (1.0 - alpha) * region, 0, 255).astype(np.uint8)
    full = np.zeros((height, width), np.uint8)
    full[dst_y0:dst_y1, dst_x0:dst_x1] = patch_mask
    return full


def _shadow(canvas: np.ndarray, mask: np.ndarray, x: int, y: int) -> None:
    height, width = canvas.shape[:2]
    crop_h, crop_w = mask.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + crop_w), min(height, y + crop_h)
    if x1 <= x0 or y1 <= y0:
        return
    src = mask[: y1 - y0, : x1 - x0]
    blurred = cv2.GaussianBlur(src, (0, 0), 3)
    alpha = (blurred.astype(np.float32) / 255.0) * 0.35
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(region * (1.0 - alpha[:, :, None]), 0, 255).astype(np.uint8)


def _rotate(
    image: np.ndarray,
    mask: np.ndarray,
    angle: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(height * sin + width * cos)
    new_h = int(height * cos + width * sin)
    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rotated_mask = cv2.warpAffine(mask, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated, rotated_mask


def _jitter(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.integers(-4, 5))) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + int(rng.integers(-12, 13)), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * float(rng.uniform(0.85, 1.15)), 0, 255)
    jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    alpha = (mask > 0)[:, :, None]
    return np.where(alpha, jittered, image)


def _fit_scale(crop: Instance, width: int, height: int, slot_count: int) -> float:
    cols = 3 if slot_count >= 9 else 2
    rows = int(np.ceil(slot_count / cols))
    cell = min(width / (cols + 1), height / (rows + 1))
    longest = max(crop.crop_mask.shape[:2])
    if longest <= 1:
        return 1.0
    return float(np.clip((cell * 0.7) / longest, 0.25, 1.4))


def _slots(pattern: str, width: int, height: int) -> list[tuple[int, int]]:
    if pattern == "circle":
        count = 8
        radius_x = width * 0.28
        radius_y = height * 0.28
        cx, cy = width / 2, height / 2
        return [
            (int(cx + radius_x * np.cos(theta)), int(cy + radius_y * np.sin(theta)))
            for theta in np.linspace(0, 2 * np.pi, count, endpoint=False)
        ]
    if pattern == "triangle":
        rows = (1, 2, 3)
        points: list[tuple[int, int]] = []
        for row, count in enumerate(rows):
            y = int(height * (0.28 + row * 0.22))
            for col in range(count):
                x = int(width * (0.5 + (col - (count - 1) / 2) * 0.18))
                points.append((x, y))
        return points
    cols, rows = (4, 2) if pattern == "rectangle" else (3, 3)
    points = []
    for row in range(rows):
        for col in range(cols):
            x = int(width * (0.18 + col * (0.64 / max(1, cols - 1))))
            y = int(height * (0.22 + row * (0.56 / max(1, rows - 1))))
            points.append((x, y))
    return points
