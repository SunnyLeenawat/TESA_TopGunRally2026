"""Train the bean autoencoder, plot the latent space, and export the encoder."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from coffee.ae.bank import good_threshold, l2_normalize, prototypes_from, save_bank
from coffee.ae.crops import composite
from coffee.config import AppConfig, ConfigError

logger = logging.getLogger("coffee.train.ae")


def train_autoencoder(cfg: AppConfig) -> Path:
    import torch
    from torch import nn

    from coffee.train.ae_net import Autoencoder

    squares, labels = _load_squares(cfg)
    train_index, val_index = _split(labels, cfg.ae_val_fraction, cfg.seed, len(cfg.classes))
    device = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    model = Autoencoder(cfg.ae_latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ae_lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_x = _batch(squares[train_index])
    train_y = labels[train_index]
    val_x = _batch(squares[val_index]) if len(val_index) else train_x[:1]
    best_loss = float("inf")
    best_state: dict | None = None
    patience = 0
    logger.info(
        "Autoencoder: %s train crops, %s val crops, latent %s, device %s",
        len(train_index),
        len(val_index),
        cfg.ae_latent_dim,
        device,
    )
    for epoch in range(1, cfg.ae_epochs + 1):
        model.train()
        order = torch.randperm(len(train_x))
        total = 0.0
        seen = 0
        for start in range(0, len(train_x), cfg.ae_batch):
            chosen = order[start : start + cfg.ae_batch]
            batch = _augment(train_x[chosen]).to(device)
            recon, _latent = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(chosen)
            seen += len(chosen)
        val_loss = _mse(model, val_x, device, loss_fn)
        logger.info("epoch %s train_mse %.5f val_mse %.5f", epoch, total / seen, val_loss)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                logger.info("Stopping early at epoch %s. Best val MSE %.5f", epoch, best_loss)
                break
    if best_state is None:
        raise ConfigError("Autoencoder training produced no checkpoint.")
    model.load_state_dict(best_state)
    model_path = cfg.model_dir / "ae.pt"
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"latent_dim": cfg.ae_latent_dim, "crop_size": cfg.ae_crop_size, "state_dict": model.state_dict()},
        model_path,
    )
    _write_recon(model, val_x, device, cfg.output_dir / "ae" / "recon.jpg")
    embeddings = _embed(model, train_x, device)
    val_embeddings = _embed(model, val_x, device) if len(val_index) else embeddings
    val_labels = labels[val_index] if len(val_index) else train_y
    _write_latent(val_embeddings, val_labels, cfg.classes, cfg.seed, cfg.output_dir / "ae" / "latent.png")
    normalized = l2_normalize(embeddings)
    centers = prototypes_from(normalized, train_y, len(cfg.classes))
    threshold = good_threshold(normalized, train_y, centers, cfg.classes.index("good"), cfg.ae_anomaly_quantile)
    bank_path = cfg.model_dir / "ae_bank.npz"
    save_bank(bank_path, normalized, train_y, centers, cfg.classes, threshold, cfg.ae_crop_size, cfg.ae_latent_dim)
    exported = _export_encoder(model.encoder.cpu().eval(), cfg)
    delta = verify_encoder(cfg, squares[:8], model.encoder)
    logger.info("Encoder parity on %s crops, max abs %.6f", min(8, len(squares)), delta)
    metrics = {
        "val_mse": round(best_loss, 6),
        "train_crops": int(len(train_index)),
        "val_crops": int(len(val_index)),
        "good_distance_threshold": threshold,
        "encoder_max_abs": delta,
        "ncnn": str(exported),
    }
    metrics_path = cfg.output_dir / "ae" / "metrics.yaml"
    metrics_path.write_text(yaml.safe_dump(metrics, sort_keys=False), encoding="utf-8")
    logger.info("Autoencoder saved. val MSE %.5f, bank %s", best_loss, bank_path)
    return model_path


def _load_squares(cfg: AppConfig) -> tuple[np.ndarray, np.ndarray]:
    root = cfg.output_dir / "crops"
    squares: list[np.ndarray] = []
    labels: list[int] = []
    for class_id, name in enumerate(cfg.classes):
        folder = root / name
        paths = sorted(folder.glob("*.png")) if folder.is_dir() else []
        if not paths:
            raise ConfigError(f"No crops for {name} in {folder}. Run prep before train-ae.")
        for path in paths:
            bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
                raise ConfigError(f"Crop is not a BGRA PNG: {path}")
            squares.append(composite(bgra, cfg.ae_crop_size))
            labels.append(class_id)
    return np.stack(squares), np.asarray(labels, dtype=np.int64)


def _split(labels: np.ndarray, fraction: float, seed: int, class_count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    for class_id in range(class_count):
        index = np.flatnonzero(labels == class_id)
        order = rng.permutation(len(index))
        holdout = 0 if len(index) < 4 or fraction <= 0 else min(len(index) - 1, max(1, round(len(index) * fraction)))
        val.extend(index[order[:holdout]].tolist())
        train.extend(index[order[holdout:]].tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def _batch(squares: np.ndarray):
    import torch

    rgb = squares[:, :, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))


def _augment(batch):
    import torch

    out = batch.clone()
    if torch.rand(1).item() < 0.5:
        out = torch.flip(out, dims=[3])
    if torch.rand(1).item() < 0.5:
        out = torch.flip(out, dims=[2])
    return out


def _mse(model, images, device, loss_fn) -> float:
    import torch

    model.eval()
    with torch.no_grad():
        recon, _latent = model(images.to(device))
        return float(loss_fn(recon, images.to(device)))


def _embed(model, images, device) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        _recon, latent = model(images.to(device))
    return latent.cpu().numpy().astype(np.float32)


def _write_recon(model, images, device, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        recon, _latent = model(images[:6].to(device))
    original = _to_bgr(images[:6])
    restored = _to_bgr(recon.cpu())
    grid = np.concatenate([original, restored], axis=1)
    if not cv2.imwrite(str(path), grid):
        raise ConfigError(f"Could not write {path}.")


def _to_bgr(chw) -> np.ndarray:
    array = chw.numpy() if hasattr(chw, "numpy") else chw
    tiles = []
    for image in array:
        bgr = np.clip(image[::-1].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        tiles.append(bgr)
    return np.concatenate(tiles, axis=0)


def _write_latent(embeddings: np.ndarray, labels: np.ndarray, class_names: tuple[str, ...], seed: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points, method = _project(embeddings, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5))
    for class_id, name in enumerate(class_names):
        chosen = labels == class_id
        if np.any(chosen):
            axis.scatter(points[chosen, 0], points[chosen, 1], s=18, label=name)
    axis.legend(loc="best", fontsize=8)
    axis.set_title(f"Encoder latent space ({method})")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    logger.info("Latent plot (%s): %s", method, path)


def _project(embeddings: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    count = len(embeddings)
    if count >= 5:
        try:
            from sklearn.manifold import TSNE

            perplexity = min(15, count - 1)
            points = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(embeddings)
            return points, "t-SNE"
        except ImportError:
            logger.warning("scikit-learn is not installed. The latent plot uses PCA.")
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T, "PCA"


def _export_encoder(encoder, cfg: AppConfig) -> Path:
    import torch

    destination = cfg.model_dir / "ae_encoder_ncnn"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    dummy = torch.zeros(1, 3, cfg.ae_crop_size, cfg.ae_crop_size)
    import pnnx

    logger.info("Exporting encoder with PNNX %s", getattr(pnnx, "__version__", "unknown"))
    pnnx.export(
        encoder,
        str(destination / "encoder.pt"),
        inputs=dummy,
        ncnnparam=str(destination / "model.ncnn.param"),
        ncnnbin=str(destination / "model.ncnn.bin"),
        ncnnpy=str(destination / "model_ncnn.py"),
        pnnxparam=str(destination / "model.pnnx.param"),
        pnnxbin=str(destination / "model.pnnx.bin"),
        pnnxpy=str(destination / "model_pnnx.py"),
        pnnxonnx=str(destination / "model.pnnx.onnx"),
        fp16=False,
        device="cpu",
    )
    for name in ("model.pnnx.param", "model.pnnx.bin", "model_pnnx.py", "model.pnnx.onnx", "encoder.pt"):
        (destination / name).unlink(missing_ok=True)
    _probe_and_write_metadata(destination, encoder, dummy, cfg)
    return destination


def _probe_and_write_metadata(destination: Path, encoder, dummy, cfg: AppConfig) -> None:
    import ncnn
    import torch

    with torch.no_grad():
        torch_out = encoder(dummy).numpy()
    net = ncnn.Net()
    net.opt.num_threads = cfg.ncnn_threads
    param = destination / "model.ncnn.param"
    if net.load_param(str(param)) != 0 or net.load_model(str(destination / "model.ncnn.bin")) != 0:
        raise ConfigError(f"NCNN could not load the exported encoder in {destination}.")
    extractor = net.create_extractor()
    tensor = np.ascontiguousarray(dummy.numpy()[0])
    extractor.input(net.input_names()[0], ncnn.Mat(tensor))
    blobs: dict[str, np.ndarray] = {}
    for name in net.output_names():
        status, mat = extractor.extract(name)
        if status != 0:
            raise ConfigError(f"NCNN extract failed for encoder blob {name}.")
        blobs[name] = np.array(mat)
    lines = [
        f"inputs={list(net.input_names())}",
        f"outputs={list(net.output_names())}",
    ]
    for name, array in blobs.items():
        lines.append(f"{name} shape={array.shape} min={float(array.min()):.6f} max={float(array.max()):.6f}")
    torch_vector = torch_out.reshape(-1)
    lines.append(f"torch_latent shape={tuple(torch_out.shape)}")
    chosen, delta = _matching_blob(blobs, torch_vector)
    lines.append(f"matched={chosen} max_abs={delta:.6f}")
    probe = cfg.output_dir / "ae" / "ncnn_io.txt"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Encoder NCNN probe:\n%s", "\n".join(lines))
    metadata = {
        "input": net.input_names()[0],
        "output": chosen,
        "latent_dim": cfg.ae_latent_dim,
        "crop_size": cfg.ae_crop_size,
        "names": {index: name for index, name in enumerate(cfg.classes)},
    }
    (destination / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")


def _matching_blob(blobs: dict[str, np.ndarray], torch_vector: np.ndarray) -> tuple[str, float]:
    best_name = ""
    best_delta = float("inf")
    for name, array in blobs.items():
        if array.size != torch_vector.size:
            continue
        delta = float(np.max(np.abs(array.reshape(-1) - torch_vector)))
        if delta < best_delta:
            best_name = name
            best_delta = delta
    if best_name == "":
        raise ConfigError(
            f"No NCNN blob has {torch_vector.size} values. Blobs: "
            + ", ".join(f"{name} {array.shape}" for name, array in blobs.items())
        )
    return best_name, best_delta


def verify_encoder(cfg: AppConfig, squares: np.ndarray, encoder) -> float:
    """Compare the PyTorch encoder with NCNN on real crops. Raises if they diverge."""
    import torch

    from coffee.infer.ncnn_encoder import NcnnEncoder

    runtime = NcnnEncoder(cfg.model_dir / "ae_encoder_ncnn", cfg.ncnn_threads, cfg.ncnn_use_vulkan)
    worst = 0.0
    encoder.eval()
    with torch.no_grad():
        for square in squares:
            tensor = _batch(square[None])
            torch_vector = l2_normalize(encoder(tensor).numpy()[0])
            ncnn_vector = runtime.embed(square)
            worst = max(worst, float(np.max(np.abs(torch_vector - ncnn_vector))))
    if worst > 1e-3:
        raise ConfigError(f"Encoder NCNN parity failed. Max abs difference {worst:.6f}.")
    return worst


def _device(name: str):
    import torch

    if str(name).isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{name}")
        logger.warning("CUDA device %s is not available. Training the autoencoder on CPU.", name)
        return torch.device("cpu")
    return torch.device(name)
