# Contest day

Train on the laptop. The Pi only runs `./run.sh`. No camera.

## The one line

`.env` line 3, `HOST_DATA_DIR`, is the dataset root Docker mounts at `/data`. It is now

```bash
HOST_DATA_DIR=/home/sunny/Documents/University/TESA/Track2/data
```

Point it at the official root, or copy the official photos into this folder. `make prep` uses `<root>/train/<class>/` when that tree exists. Class folders, in order: `good`, `shell`, `full_black`, `hull_husk`, `broken`, `parchment`, `immature`. Test photos stay out of training: pass them only as `--input` on the Pi. After a retrain, `make pi-bundle` reads weights from `$HOST_DATA_DIR/models`.

## Each day

Laptop, from this repo. Budget about 45 minutes if the Pi image was built the night before. Add about 2 minutes per new training photo for bean extraction. Training itself is about 2 minutes, export about 1 minute.

```bash
docker context use default          # once on this machine
make prep && make train-yolo && make export && make train-ae && make pi-bundle
scp dist/coffee-edge-pi.tar.gz <pi-user>@<pi-host>:
```

Pi, after `tar -xzf coffee-edge-pi.tar.gz && cd coffee-edge-pi`. A folder of test photos is `~/images`. Re-copy the tarball only when the weights changed. A new tarball does not rebuild the image if the code is unchanged.

| Day | Laptop | Pi | Hand in |
| --- | --- | --- | --- |
| 1 | commands above | `./run.sh 1 ~/images ~/out-day1` | `predictions.csv`, one class per image |
| 2 | retrain only if new mask labels arrived | `./run.sh 2 ~/images ~/out-day2` | `overlays/*.jpg`, box and mask |
| 3 | retrain only if the mixed train set arrived | `./run.sh 3 ~/images ~/out-day3` | `predictions.json`: `counts`, `defect_ratio`, `per_bean`, `latency_ms`, plus overlays |

`latency_ms` is the Pi time. Pi 5 at 640 is about 0.26 s per image in the brief. The first image is slower.

## If something is wrong

- Pi is too slow. On the laptop run `COFFEE_IMGSZ=320 make export && make pi-bundle`, scp again. The NCNN file is fixed at the exported size, so do not change `imgsz` only on the Pi. Brief timing at 320 is about 0.08 s.
- Classes look random on piles. Run `make synth && make train-yolo && make export && make pi-bundle`. Synthetic piles are added to train only. Raise `synth.images_per_class` in `config/config.yaml` for more of them.
- Leave `autoencoder.fusion_enabled` false. On the mock val set, YOLO alone was 0.691 and YOLO+AE was 0.632.
- NCNN fails to load. Re-run `make export` on the laptop. This bundle has no ONNX runtime on the Pi.
