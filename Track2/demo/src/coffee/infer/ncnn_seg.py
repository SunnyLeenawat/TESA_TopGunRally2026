"""YOLO26n-seg decoder for the NCNN export produced by Ultralytics 8.4.212.

Re-measured on the Day 3 export (imgsz 640, ``end2end: false``, registry class order),
not assumed from YOLOv8. See ``inspect.json`` next to the NCNN files.

- input blob ``in0``: float32 RGB, shape (3, imgsz, imgsz), values 0 to 1
- output blob ``out0``: shape (43, 8400) = (4 + 7 classes + 32 mask channels, anchors)
  channels are center-xywh in letterbox pixels, sigmoid class scores, mask coefficients
- output blob ``out1``: mask prototypes, shape (32, 160, 160)

An export with ``end2end: true`` or any other blob names is rejected.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from coffee.config import AppConfig, ConfigError

logger = logging.getLogger("coffee.infer")

_MAX_DET = 300
_CLASS_OFFSET = 7680.0


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    box_xyxy: np.ndarray
    mask: np.ndarray
    input_mask: np.ndarray | None = None


@dataclass(frozen=True)
class Letterbox:
    image: np.ndarray
    gain_x: float
    gain_y: float
    pad_x: int
    pad_y: int
    resized_width: int
    resized_height: int


def letterbox(image: np.ndarray, size: int, pad_value: int = 114) -> Letterbox:
    """Match Ultralytics ``LetterBox`` for a square target: center pad, value 114."""
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width = max(round(width * scale), 1)
    resized_height = max(round(height * scale), 1)
    pad_w = (size - resized_width) / 2
    pad_h = (size - resized_height) / 2
    pad_x = round(pad_w - 0.1)
    pad_y = round(pad_h - 0.1)
    right = round(pad_w + 0.1)
    bottom = round(pad_h + 0.1)
    resized = image
    if (width, height) != (resized_width, resized_height):
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, pad_y, bottom, pad_x, right, cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value)
    )
    return Letterbox(
        image=padded,
        gain_x=resized_width / width,
        gain_y=resized_height / height,
        pad_x=pad_x,
        pad_y=pad_y,
        resized_width=resized_width,
        resized_height=resized_height,
    )


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def class_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_det: int = _MAX_DET,
) -> np.ndarray:
    """Class-aware greedy NMS. Same-class boxes are separated the way Ultralytics offsets them."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    shifted = boxes.copy()
    offset = class_ids.astype(np.float32) * _CLASS_OFFSET
    shifted[:, 0] += offset
    shifted[:, 2] += offset
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size and len(kept) < max_det:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[_iou_one_to_many(shifted[current], shifted[rest]) <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def inspect_export(model_dir: Path, cfg: AppConfig) -> dict[str, object]:
    """Print the real NCNN blob names and shapes. Refuse a layout this decoder has not measured."""
    metadata_path = model_dir / "metadata.yaml"
    if not metadata_path.is_file():
        raise ConfigError(f"Missing {metadata_path}.")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    end2end = bool(metadata.get("end2end"))
    names = tuple(metadata["names"][index] for index in range(len(metadata["names"])))
    logger.info(
        "NCNN metadata end2end=%s imgsz=%s names=%s",
        end2end,
        metadata.get("imgsz"),
        names,
    )
    if end2end:
        raise ConfigError("NCNN export has end2end: true. Do not guess an end-to-end layout.")
    if names != cfg.classes:
        raise ConfigError(f"Exported names {names} do not match the class registry {cfg.classes}.")
    import ncnn

    net = ncnn.Net()
    net.opt.num_threads = cfg.ncnn_threads
    param = model_dir / "model.ncnn.param"
    weights = model_dir / "model.ncnn.bin"
    if net.load_param(str(param)) != 0 or net.load_model(str(weights)) != 0:
        raise ConfigError(f"NCNN could not load {param}.")
    input_name = net.input_names()[0]
    output_names = tuple(net.output_names())
    sample = _inspect_image(cfg, cfg.imgsz)
    tensor = _as_input(letterbox(sample, cfg.imgsz).image)
    extractor = net.create_extractor()
    extractor.input(input_name, ncnn.Mat(tensor))
    shapes: dict[str, list[int]] = {}
    for name in output_names:
        status, mat = extractor.extract(name)
        if status != 0:
            raise ConfigError(f"NCNN extract failed for blob {name}.")
        shapes[name] = [int(value) for value in np.array(mat).shape]
    logger.info("NCNN blobs input=%s outputs=%s shapes=%s", input_name, output_names, shapes)
    if set(output_names) != {"out0", "out1"}:
        raise ConfigError(
            f"Measured output blobs {output_names}. The decoder expects out0 and out1. "
            "Update the decoder from these shapes before parity."
        )
    report = {
        "end2end": end2end,
        "input": input_name,
        "outputs": list(output_names),
        "shapes": shapes,
        "names": list(names),
        "imgsz": [int(value) for value in metadata["imgsz"]],
    }
    destination = model_dir / "inspect.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", destination)
    return report


def _inspect_image(cfg: AppConfig, size: int) -> np.ndarray:
    folder = cfg.output_dir / "day3" / "yolo_roi" / "images" / "val"
    if folder.is_dir():
        images = sorted(folder.glob("*.jpg"))
        if images:
            image = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
            if image is not None:
                return image
    return np.zeros((size, size, 3), np.uint8)


def ncnn_model_dir(model_dir: Path, output_dir: Path) -> Path:
    stable = model_dir / "yolo26n-seg_ncnn"
    if (stable / "model.ncnn.param").is_file():
        return stable
    fallback = output_dir / "runs" / "yolo26n-seg" / "weights" / "best_ncnn_model"
    if (fallback / "model.ncnn.param").is_file():
        return fallback
    raise ConfigError(f"No NCNN model in {stable} or {fallback}. Run export first.")


class NcnnSegmenter:
    def __init__(
        self,
        model_dir: Path,
        class_names: tuple[str, ...],
        imgsz: int,
        threads: int,
        conf: float,
        iou: float,
        use_vulkan: bool = False,
    ) -> None:
        import ncnn

        metadata_path = model_dir / "metadata.yaml"
        if not metadata_path.is_file():
            raise ConfigError(f"Missing {metadata_path}.")
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("end2end"):
            raise ConfigError(
                "This NCNN export has end2end: true. The decoder only accepts the dense head "
                "(end2end: false) that this training export produced."
            )
        names = tuple(metadata["names"][index] for index in range(len(metadata["names"])))
        if names != class_names:
            raise ConfigError(f"Model classes {names} do not match config classes {class_names}.")
        exported_size = tuple(int(v) for v in metadata["imgsz"])
        if exported_size != (imgsz, imgsz):
            raise ConfigError(f"Model was exported at {exported_size}, config imgsz is {imgsz}. Re-export.")
        self._ncnn = ncnn
        self._names = names
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._net = ncnn.Net()
        self._net.opt.num_threads = threads
        self._net.opt.use_vulkan_compute = bool(use_vulkan)
        param = model_dir / "model.ncnn.param"
        weights = model_dir / "model.ncnn.bin"
        if self._net.load_param(str(param)) != 0 or self._net.load_model(str(weights)) != 0:
            raise ConfigError(f"NCNN could not load {param}.")
        self._input = self._net.input_names()[0]
        self._outputs = tuple(self._net.output_names())
        logger.info(
            "NCNN model %s input %s outputs %s threads %s",
            model_dir,
            self._input,
            self._outputs,
            threads,
        )

    def predict(self, image_bgr: np.ndarray, conf: float | None = None) -> list[Detection]:
        boxed = letterbox(image_bgr, self._imgsz)
        tensor = _as_input(boxed.image)
        prediction, prototypes = self._forward(tensor)
        return _decode(
            prediction,
            prototypes,
            boxed,
            image_bgr.shape[:2],
            self._conf if conf is None else conf,
            self._iou,
            len(self._names),
        )

    def _forward(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        extractor = self._net.create_extractor()
        extractor.input(self._input, self._ncnn.Mat(tensor))
        blobs: dict[str, np.ndarray] = {}
        for name in self._outputs:
            status, mat = extractor.extract(name)
            if status != 0:
                raise ConfigError(f"NCNN extract failed for blob {name}.")
            blobs[name] = np.array(mat)
        if "out0" not in blobs or "out1" not in blobs:
            raise ConfigError(f"Expected blobs out0 and out1, got {sorted(blobs)}.")
        return blobs["out0"], blobs["out1"]


def _as_input(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)


def _decode(
    prediction: np.ndarray,
    prototypes: np.ndarray,
    boxed: Letterbox,
    original_hw: tuple[int, int],
    conf: float,
    iou: float,
    class_count: int,
) -> list[Detection]:
    mask_dim = int(prototypes.shape[0])
    channels = 4 + class_count + mask_dim
    if prediction.shape[0] != channels and prediction.shape[-1] == channels:
        prediction = prediction.T
    if prediction.shape[0] != channels:
        raise ConfigError(
            f"out0 has shape {prediction.shape}, expected ({channels}, anchors) "
            f"for {class_count} classes and {mask_dim} mask channels."
        )
    boxes = xywh_to_xyxy(prediction[:4].T)
    class_scores = prediction[4 : 4 + class_count]
    coefficients = prediction[4 + class_count :].T
    best = class_scores.max(axis=0)
    keep = best >= conf
    if not np.any(keep):
        return []
    boxes = boxes[keep]
    scores = best[keep]
    class_ids = class_scores.argmax(axis=0)[keep].astype(np.int64)
    coefficients = coefficients[keep]
    chosen = class_nms(boxes, scores, class_ids, iou)
    boxes = boxes[chosen]
    scores = scores[chosen]
    class_ids = class_ids[chosen]
    coefficients = coefficients[chosen]
    letter_masks = _masks(coefficients, prototypes, boxes, boxed.image.shape[:2])
    original = _scale_boxes(boxes, boxed, original_hw)
    detections: list[Detection] = []
    for index in range(len(boxes)):
        detections.append(
            Detection(
                class_id=int(class_ids[index]),
                score=float(scores[index]),
                box_xyxy=original[index],
                mask=_mask_on_original(letter_masks[index], boxed, original_hw),
                input_mask=letter_masks[index],
            )
        )
    return detections


def _masks(coefficients: np.ndarray, prototypes: np.ndarray, boxes: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear-upsample logits, threshold at 0, then crop to each box. Same order as Ultralytics."""
    mask_dim, proto_h, proto_w = prototypes.shape
    logits = coefficients @ prototypes.reshape(mask_dim, -1)
    logits = logits.reshape(-1, proto_h, proto_w)
    height, width = shape_hw
    upsampled = np.empty((len(logits), height, width), dtype=np.float32)
    for index, mask in enumerate(logits):
        upsampled[index] = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    binary = (upsampled > 0).astype(np.uint8)
    return _crop_masks(binary, boxes)


def _crop_masks(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    count, height, width = masks.shape
    columns = np.arange(width, dtype=np.float32)[None, None, :]
    rows = np.arange(height, dtype=np.float32)[None, :, None]
    x1 = boxes[:, 0][:, None, None]
    y1 = boxes[:, 1][:, None, None]
    x2 = boxes[:, 2][:, None, None]
    y2 = boxes[:, 3][:, None, None]
    cropped = masks.copy()
    cropped *= (columns >= x1) & (columns < x2)
    cropped *= (rows >= y1) & (rows < y2)
    return cropped


def _scale_boxes(boxes: np.ndarray, boxed: Letterbox, original_hw: tuple[int, int]) -> np.ndarray:
    scaled = boxes.copy()
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - boxed.pad_x) / boxed.gain_x
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - boxed.pad_y) / boxed.gain_y
    height, width = original_hw
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, width)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, height)
    return scaled


def _mask_on_original(mask: np.ndarray, boxed: Letterbox, original_hw: tuple[int, int]) -> np.ndarray:
    cropped = mask[boxed.pad_y : boxed.pad_y + boxed.resized_height, boxed.pad_x : boxed.pad_x + boxed.resized_width]
    height, width = original_hw
    if cropped.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    restored = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_NEAREST)
    return (restored > 0).astype(np.uint8) * 255


def _iou_one_to_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_others = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(0.0, others[:, 3] - others[:, 1])
    union = area_box + area_others - intersection
    return np.where(union > 0, intersection / union, 0.0)
