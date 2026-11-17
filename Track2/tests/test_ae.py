from __future__ import annotations

import numpy as np

from coffee.ae.bank import Bank, classify, l2_normalize, prototypes_from
from coffee.ae.crops import composite
from coffee.ae.fuse import fuse

_NAMES = ("good", "shell", "full_black", "hull_husk", "broken", "parchment", "immature")


def test_composite_centers_an_opaque_crop_on_gray() -> None:
    bgra = np.zeros((4, 8, 4), np.uint8)
    bgra[:, :, 2] = 255
    bgra[:, :, 3] = 255
    square = composite(bgra, 16)
    assert square.shape == (16, 16, 3)
    assert square[0, 0, 0] == 114
    assert int(square[8, 8, 2]) > 200


def test_knn_and_prototype_follow_the_nearest_class() -> None:
    embeddings = l2_normalize(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], np.float32))
    labels = np.array([0, 0, 1, 1])
    centers = prototypes_from(embeddings, labels, 2)
    bank = Bank(embeddings, labels, centers, ("good", "shell"), 0, 0.5, 64, 2)
    knn_class, share, proto_class, _anomaly = classify(np.array([0.0, 1.0], np.float32), bank, k=3)
    assert knn_class == 1
    assert proto_class == 1
    assert share >= 0.6


def test_fusion_off_keeps_the_yolo_class() -> None:
    bank = _bank()
    result = fuse(0, 0.2, np.array([0.0, 1.0], np.float32), bank, enabled=False, k=1, knn_share=0.6, low_conf=0.45, confusable=(("good", "shell"),))
    assert result.class_id == 0
    assert result.changed is False


def test_fusion_flips_a_low_confidence_confusable_pair() -> None:
    bank = _bank()
    result = fuse(0, 0.3, np.array([0.0, 1.0], np.float32), bank, enabled=True, k=1, knn_share=0.6, low_conf=0.45, confusable=(("good", "shell"),))
    assert result.class_id == 1
    assert result.changed is True


def _bank() -> Bank:
    embeddings = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]], np.float32))
    labels = np.array([0, 1])
    centers = prototypes_from(embeddings, labels, 2)
    return Bank(embeddings, labels, centers, ("good", "shell"), 0, 0.05, 64, 2)
