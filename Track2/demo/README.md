# Coffee-edge

TESA Top Gun 2026 Track 2, Day 3. Train on this laptop. Run inference on the Raspberry Pi with NCNN. No camera. No PyTorch on the Pi.

The 15-minute contest steps are in `RUNBOOK.md`. The hand-in zip is `dist/team_55_Day3.zip`.

## Do not rerun the old rehearsal

`make mock-data`, `make prep`, `make synth`, `make train-yolo`, `make export`, `make train-ae`, `make ablate`, `make infer-day1`, `make infer-day2`, and `make infer-day3` belong to an earlier dataset and class list. Running them again replaces `data/models/yolo26n-seg_ncnn`, which is the model inside the submission zip.

## Dataset and classes

Edit `HOST_DATA_DIR` in `.env`. Compose mounts that folder at `/data`. Class ids and the seven submission names are in `config/classes.yaml`. Older folder names `full_black` and `hull_husk` alias to `black` and `husk`.

The 16 mixed photos and `labels.json` are in `data/mix`. Judge HTTP and MQTT values live in `.env`. That file is gitignored. `COFFEE_HTTP_TOKEN` is the website password. `COFFEE_MQTT_PASSWORD` is the MQTT key. Do not swap them. Leave `COFFEE_TEAM=team_55`. The zip name comes from that variable.

## Laptop commands that match this Day 3 model

```bash
make test
make day3-data     # labels.json -> YOLO labels, image split, crops
make day3-s5       # gray-paper piles, retrain, export NCNN, sweep thresholds
make day3-infer    # write seg_mask, predictions.json, mqtt, vis
make day3-score    # local score on data/mix
make day3-mock     # local mock only; it is not the judges' API
make day3-ae       # optional encoder; fusion stays off
make pi-bundle     # dist/coffee-edge-pi.tar.gz
make submission    # dist/team_55_day3/ and dist/team_55_Day3.zip
```

`make day3-s5` is the training run whose weights are shipped. Confidence `0.05`, dedup `0.5`, refinement off, autoencoder fusion off. On the 16 mixed photos the class-aware score was 46.46 / 60 and the class-ignored score was 54.02 / 60. The median on this laptop was 1.428 s. That number is not the Pi time.

## Judges' server

`client.py` talks to `https://edgeai-tesa.c333m.site`. Each round is one `GET /api/v1/image?mode=train`, then inference, then a JSON `POST /api/v1/submit` and an MQTT publish at the same time. The POST body is `{img_id, predictions}`. HTTP 409 means that `img_id` was already sent. HTTP 403 ends the loop.

MQTT is WebSocket TLS on `tesa-score.c333m.site:443`, path `/mqtt/`, client id `TGR-55:device-1`. The train-day topic is `/event/<MQTT key>/EdgeAI/system-testing`. `EdgeAI` is case-sensitive. A team login can publish. MQTTX subscribe with that login returns Not authorized (135). Do not open MQTTX with the same client id while the Pi is connected.

Stay on `mode=train`. `mode=test` spends the five-image quota. Set `COFFEE_HTTP_MODE=test` only on the scored day.

One practice image, without spending test quota:

```bash
.venv/bin/python client.py --weights data/models/yolo26n-seg_ncnn --out /tmp/day3-client --limit 1
```

The image is kept at `/tmp/day3-client/rounds/<img_id>/image/<X-Image-Name>`. The server does not send ground-truth labels. `incoming/` holds only the latest download and is overwritten when the next file reuses the same name. Omit `--limit` only when the feed should run until HTTP 403.

## Pi

Copy `dist/team_55_Day3.zip` before the window. Inside it:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python bench_pi.py --input ~/images --weights weights --out ~/out-warmup
.venv/bin/python infer.py --input <test_folder> --weights weights --out ~/out-day3
```

`infer.py` must log `Output contract passed`. If Docker is down, `./run.sh 3` from `dist/coffee-edge-pi` uses the same venv fallback. Time the pass with `bench_pi.py` in the venv. Docker and QEMU times are not the contest latency.

## GPU

The training container requests `nvidia.com/gpu=all`. On this Docker daemon, `gpus: all` fails with `AMD CDI spec not found`. A training container should report `torch.cuda.is_available() == True` and the RTX 4060.
