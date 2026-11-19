The contest sheet is `RUNBOOK.md`. Train on the laptop, then run `./run.sh` on the Pi. There is no camera.

**The one line.** `.env` line 3 is the dataset root:

```bash
HOST_DATA_DIR=/home/sunny/Documents/University/TESA/Track2/data
```

Point that at the official root, or copy the official photos into this folder. `make prep` uses `<root>/train/<class>/` when that tree exists. Class folders, in order: `good`, `shell`, `full_black`, `hull_husk`, `broken`, `parchment`, `immature`. Test photos are not part of training. After a retrain, `make pi-bundle` reads weights from `$HOST_DATA_DIR/models`.

**Each day.** Budget about 45 minutes if the Pi image was built the night before, plus about 2 minutes per new training photo for extraction. Training is about 2 minutes and export about 1 minute.

```bash
docker context use default
make prep && make train-yolo && make export && make train-ae && make pi-bundle
scp dist/coffee-edge-pi.tar.gz <pi-user>@<pi-host>:
```

On the Pi, after `tar -xzf coffee-edge-pi.tar.gz && cd coffee-edge-pi`:

| Day | Laptop | Pi | Hand in |
| --- | --- | --- | --- |
| 1 | the commands above | `./run.sh 1 ~/images ~/out-day1` | `predictions.csv`, one class per image |
| 2 | retrain only if new mask labels arrived | `./run.sh 2 ~/images ~/out-day2` | `overlays/*.jpg`, box and mask |
| 3 | retrain only if the mixed train set arrived | `./run.sh 3 ~/images ~/out-day3` | `predictions.json` (`counts`, `defect_ratio`, `per_bean`, `latency_ms`) plus overlays |

`latency_ms` is the Pi time. The brief's Pi 5 figure at 640 is about 0.26 s per image. The first image is slower. Copy a new tarball only when the weights changed. The image itself stays cached.

**If something is wrong**

- The Pi is too slow: on the laptop run `COFFEE_IMGSZ=320 make export && make pi-bundle` and copy the tarball again. The NCNN file is fixed at the exported size, so do not change `imgsz` only on the Pi. The brief's figure at 320 is about 0.08 s.
- Piles look nothing like the neat training photos: `make synth && make train-yolo && make export && make pi-bundle`. Those synthetic piles go into train only. Raise `synth.images_per_class` in `config/config.yaml` for more of them.
- Leave `autoencoder.fusion_enabled` false. On the mock validation set, YOLO alone was 0.691 and YOLO plus the autoencoder was 0.632.
- NCNN fails to load: run `make export` again on the laptop. This bundle has no ONNX runtime on the Pi.

---

No code changes. Paths are not written in the Python files.

**Training photos** go under the folder named by `HOST_DATA_DIR` in `.env`. That line is already:

```bash
HOST_DATA_DIR=/home/sunny/Documents/University/TESA/Track2/data
```

Leave it, and copy the official photos here:

```text
data/train/good/
data/train/shell/
data/train/full_black/
data/train/hull_husk/
data/train/broken/
data/train/parchment/
data/train/immature/
```

As soon as `data/train/` exists, `make prep` uses it and ignores `data/mock/`. You can instead point `HOST_DATA_DIR` at another folder; that folder must contain the same `train/<class>/` layout. Docker mounts it at `/data`, and `make pi-bundle` reads weights from `$HOST_DATA_DIR/models` after training.

**Test photos** never go in `train/`. On the Pi they are only the `--input` argument:

```bash
./run.sh 1 ~/images ~/out-day1
./run.sh 2 ~/images ~/out-day2
./run.sh 3 ~/images ~/out-day3
```

`~/images` can be one photo or a folder of photos. A folder of class subfolders is also fine. Nothing in the source has to be edited for either set.
