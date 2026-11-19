"""Class prototypes and a numpy kNN over encoder embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coffee.config import ConfigError


@dataclass(frozen=True)
class Bank:
    embeddings: np.ndarray
    labels: np.ndarray
    prototypes: np.ndarray
    class_names: tuple[str, ...]
    good_index: int
    good_threshold: float
    crop_size: int
    latent_dim: int


def save_bank(
    path: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    class_names: tuple[str, ...],
    good_threshold: float,
    crop_size: int,
    latent_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        prototypes=np.asarray(prototypes, dtype=np.float32),
        class_names=np.asarray(class_names),
        good_threshold=np.float32(good_threshold),
        crop_size=np.int64(crop_size),
        latent_dim=np.int64(latent_dim),
    )


def load_bank(path: Path) -> Bank:
    if not path.is_file():
        raise ConfigError(f"Missing embedding bank {path}. Run train-ae first.")
    with np.load(path, allow_pickle=False) as data:
        names = tuple(str(item) for item in data["class_names"])
        if "good" not in names:
            raise ConfigError("Embedding bank has no 'good' class.")
        return Bank(
            embeddings=np.asarray(data["embeddings"], dtype=np.float32),
            labels=np.asarray(data["labels"], dtype=np.int64),
            prototypes=np.asarray(data["prototypes"], dtype=np.float32),
            class_names=names,
            good_index=names.index("good"),
            good_threshold=float(data["good_threshold"]),
            crop_size=int(data["crop_size"]),
            latent_dim=int(data["latent_dim"]),
        )


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    flat = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    out = flat / np.maximum(norms, 1e-8)
    if vectors.ndim == 1:
        return out[0]
    return out


def prototypes_from(embeddings: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    means = np.zeros((class_count, embeddings.shape[1]), dtype=np.float32)
    for class_id in range(class_count):
        chosen = embeddings[labels == class_id]
        if len(chosen) == 0:
            raise ConfigError(f"No embeddings for class id {class_id}.")
        means[class_id] = chosen.mean(axis=0)
    return l2_normalize(means)


def good_threshold(embeddings: np.ndarray, labels: np.ndarray, prototypes: np.ndarray, good_index: int, quantile: float) -> float:
    good = embeddings[labels == good_index]
    distances = ((good - prototypes[good_index]) ** 2).sum(axis=1)
    return float(np.quantile(distances, quantile))


def classify(embedding: np.ndarray, bank: Bank, k: int) -> tuple[int, float, int, float]:
    """Return kNN class, kNN vote share, prototype class, and distance to the good prototype."""
    query = l2_normalize(embedding)
    distances = ((bank.embeddings - query) ** 2).sum(axis=1)
    neighbors = min(k, len(distances))
    nearest = np.argpartition(distances, neighbors - 1)[:neighbors]
    votes = np.bincount(bank.labels[nearest], minlength=len(bank.class_names)).astype(np.float64)
    knn_class = int(votes.argmax())
    share = float(votes[knn_class] / votes.sum())
    proto_distances = ((bank.prototypes - query) ** 2).sum(axis=1)
    return knn_class, share, int(proto_distances.argmin()), float(proto_distances[bank.good_index])
