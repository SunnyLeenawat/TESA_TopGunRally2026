from __future__ import annotations

import numpy as np

from coffee.infer.ncnn_seg import class_nms, letterbox, xywh_to_xyxy


def test_letterbox_pads_a_wide_image_to_square() -> None:
    image = np.zeros((640, 960, 3), np.uint8)
    boxed = letterbox(image, 640)
    assert boxed.image.shape == (640, 640, 3)
    assert boxed.pad_y > 0
    assert boxed.pad_x == 0
    assert boxed.resized_width == 640


def test_nms_keeps_separate_classes_and_drops_the_overlap() -> None:
    boxes = xywh_to_xyxy(np.array([[10, 10, 8, 8], [10.5, 10.5, 8, 8], [40, 40, 6, 6]], np.float32))
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    classes = np.array([1, 1, 2])
    kept = class_nms(boxes, scores, classes, iou_threshold=0.5)
    assert set(kept.tolist()) == {0, 2}
