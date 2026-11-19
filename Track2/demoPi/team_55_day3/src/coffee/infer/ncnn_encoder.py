"""NCNN encoder for the convolutional autoencoder.

Inspected export (PNNX 20260526, crop 64, latent 32, fp32): input blob ``in0``
is float32 RGB CHW in [0, 1], and output blob ``out0`` is the latent vector
with shape (32,). The torch encoder and this blob matched to 0 on the probe.
Blob names come from metadata.yaml written at export time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from coffee.ae.bank import l2_normalize
from coffee.config import ConfigError

logger = logging.getLogger("coffee.infer.encoder")


class NcnnEncoder:
    def __init__(self, model_dir: Path, threads: int, use_vulkan: bool = False) -> None:
        import ncnn

        metadata_path = model_dir / "metadata.yaml"
        if not metadata_path.is_file():
            raise ConfigError(f"Missing {metadata_path}.")
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        self.input_name = str(metadata["input"])
        self.output_name = str(metadata["output"])
        self.latent_dim = int(metadata["latent_dim"])
        self.crop_size = int(metadata["crop_size"])
        self._ncnn = ncnn
        self._net = ncnn.Net()
        self._net.opt.num_threads = threads
        self._net.opt.use_vulkan_compute = bool(use_vulkan)
        param = model_dir / "model.ncnn.param"
        weights = model_dir / "model.ncnn.bin"
        if self._net.load_param(str(param)) != 0 or self._net.load_model(str(weights)) != 0:
            raise ConfigError(f"NCNN could not load the encoder in {model_dir}.")
        logger.info(
            "NCNN encoder %s input %s output %s latent %s",
            model_dir,
            self.input_name,
            self.output_name,
            self.latent_dim,
        )

    def embed(self, square_bgr: np.ndarray) -> np.ndarray:
        if square_bgr.shape[0] != self.crop_size or square_bgr.shape[1] != self.crop_size:
            raise ConfigError(
                f"Encoder expects a {self.crop_size}x{self.crop_size} crop, got {square_bgr.shape[:2]}."
            )
        rgb = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        extractor = self._net.create_extractor()
        extractor.input(self.input_name, self._ncnn.Mat(tensor))
        status, mat = extractor.extract(self.output_name)
        if status != 0:
            raise ConfigError(f"NCNN extract failed for encoder blob {self.output_name}.")
        vector = np.array(mat).reshape(-1)
        if vector.size != self.latent_dim:
            raise ConfigError(f"Encoder blob has {vector.size} values, metadata latent_dim is {self.latent_dim}.")
        return l2_normalize(vector.astype(np.float32))
