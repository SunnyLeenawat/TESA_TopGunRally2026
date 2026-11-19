from __future__ import annotations

import numpy as np

from coffee.data.piles import class_weights, paste_sprite


def test_rare_class_gets_the_higher_weight() -> None:
    weights = class_weights({"good": 20, "immature": 4, "empty": 0})
    assert weights["immature"] > weights["good"]
    assert weights["empty"] == 0
    assert abs(sum(weights.values()) - 1) < 1e-6


def test_later_sprite_covers_the_overlap() -> None:
    image = np.full((30, 40, 3), 210, np.uint8)
    owner = np.zeros((30, 40), np.int16)
    first = np.full((12, 12, 3), 20, np.uint8)
    second = np.full((12, 12, 3), 90, np.uint8)
    mask = np.full((12, 12), 255, np.uint8)
    assert paste_sprite(image, owner, first, mask, (4, 4), 1) > 0
    assert paste_sprite(image, owner, second, mask, (10, 4), 2) > 0
    assert int(owner[8, 8]) == 1
    assert int(owner[8, 16]) == 2
    assert image[8, 16, 0] == 90
