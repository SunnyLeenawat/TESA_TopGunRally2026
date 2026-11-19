"""Fuse a YOLO class with the encoder bank.

Fusion stays off when the config switch is off. When it is on, the encoder vote
replaces YOLO only if the kNN and the class prototype agree, and either the
YOLO score is low, the two classes are a configured confusable pair, or YOLO
said "good" while the crop sits outside the good cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coffee.ae.bank import Bank, classify


@dataclass(frozen=True)
class FuseResult:
    class_id: int
    score: float
    anomaly: float
    changed: bool


def fuse(
    yolo_class: int,
    yolo_score: float,
    embedding: np.ndarray,
    bank: Bank,
    *,
    enabled: bool,
    k: int,
    knn_share: float,
    low_conf: float,
    confusable: tuple[tuple[str, str], ...],
) -> FuseResult:
    knn_class, share, proto_class, anomaly = classify(embedding, bank, k)
    if not enabled:
        return FuseResult(yolo_class, yolo_score, anomaly, changed=False)
    agreed = knn_class == proto_class and knn_class != yolo_class and share >= knn_share
    pair = (bank.class_names[yolo_class], bank.class_names[knn_class])
    confused = pair in confusable or (pair[1], pair[0]) in confusable
    good_but_far = yolo_class == bank.good_index and anomaly > bank.good_threshold and knn_class != bank.good_index
    if agreed and (yolo_score < low_conf or confused or good_but_far):
        return FuseResult(knn_class, share, anomaly, changed=True)
    return FuseResult(yolo_class, yolo_score, anomaly, changed=False)
