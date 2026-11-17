# Coffee-edge

TESA Top Gun 2026 Track 2 rehearsal repo. Train on the host, infer on the Pi. The contest-day steps are in `RUNBOOK.md`.

## Dataset switch

Edit the single line `HOST_DATA_DIR` in `.env`. Compose mounts that folder at `/data`. Class folders must be named `good`, `shell`, `full_black`, `hull_husk`, `broken`, `parchment`, `immature`.

## Commands

```bash
docker context use default   # once; the desktop socket is not running
make test
make hello
make hello-override
make build-train
```

`make hello` must log `data_dir=/data exists=True` and list `_sample`. `make hello-override` must log `day=2` and `data_dir=/tmp/coffee-override`.

## GPU

The training service requests the GPU with the CDI device `nvidia.com/gpu=all`. On this Docker daemon, `gpus: all` fails with `AMD CDI spec not found`; the CDI device name is the one that works. A training container should report `torch.cuda.is_available() == True` and the RTX 4060.

## Phase 1 data

```bash
make test
make mock-data
make prep
make synth
```

`make mock-data` reads `data/_sample`, extracts beans, and writes a rehearsal dataset under `data/mock/`. `make prep` checks that tree and writes `data/outputs/yolo/`. `make synth` adds touching piles under `data/outputs/synth/`. Previews are `data/outputs/preview/`.

## Phase 2 detector

```bash
make train-yolo
make export
make parity
```

`make train-yolo` fine-tunes `yolo26n-seg.pt` on the prep split plus the synthetic piles. `make export` writes `data/models/yolo26n-seg_ncnn/`. `make parity` runs the same images through PyTorch and the NCNN decoder and must log `Parity passed`.

## Phase 3 autoencoder

```bash
make train-ae
make ablate
```

`make train-ae` trains on `data/outputs/crops/`, writes `data/outputs/ae/latent.png`, and exports `data/models/ae_encoder_ncnn/` plus `data/models/ae_bank.npz`. `make ablate` logs YOLO accuracy and YOLO+AE accuracy on the val split. Fusion stays off unless `autoencoder.fusion_enabled` is true.

## Phase 4 inference

```bash
make infer-day1
make infer-day2
make infer-day3
```

These run in the Pi image. Day 1 writes `data/outputs/infer/day1/predictions.csv`. Day 2 writes overlays. Day 3 writes `predictions.json` with per-class counts, `defect_ratio`, `per_bean`, and `latency_ms`, plus overlays. Point `--input` at the contest folder when it arrives.

## Labels

The only label format is YOLO segmentation polygon text: `class_id x1 y1 x2 y2 ...` with normalized coordinates, one `.txt` per image.
