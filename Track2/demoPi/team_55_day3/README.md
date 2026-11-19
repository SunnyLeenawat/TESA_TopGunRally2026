# Day 3, 15 minutes

Team 55. Pi 5, venv, NCNN. No camera. No PyTorch on the Pi.

## 0:00–2:00 Preflight

Pi is cool. Then:

```bash
vcgencmd get_throttled    # must print throttled=0x0
vcgencmd measure_temp
df -h .                   # need a few GB free
date                      # clock is correct
```

Free RAM stays above 1500 MB. Weights folder contains `model.ncnn.param` and `best.pt`. `.env` has the HTTP base, MQTT host, port, topic, username, and password from the judges. `autoencoder.fusion_enabled` is false. `day3.refine` is false. `day3.infer_conf` is 0.05.

## 2:00–4:00 Warmup

In the submission folder, venv already installed:

```bash
.venv/bin/python bench_pi.py --input ~/images --weights weights --out ~/out-warmup
```

Read the median. Ignore the warmup image. If `get_throttled` is not `0x0` after this, cool the Pi and do not start the scored run yet.

## 4:00–11:00 Scored infer

```bash
.venv/bin/python infer.py --input <test_folder> --weights weights --out ~/out-day3
```

Check `~/out-day3` has `seg_mask/`, `predictions.json`, `mqtt/`, and `vis/`. The log must say `Output contract passed`.

## 11:00–14:00 Hand-off

If the judges' URLs are in `.env`:

```bash
.venv/bin/python client.py --weights weights --out ~/out-client
```

If `COFFEE_HTTP_BASE` is still empty, do not guess. Hand in `~/out-day3` and the zip.

## What to edit

Only `.env` lines `COFFEE_HTTP_BASE`, `COFFEE_HTTP_TOKEN`, `COFFEE_MQTT_HOST`, `COFFEE_MQTT_PORT`, `COFFEE_MQTT_TOPIC`, `COFFEE_MQTT_USERNAME`, `COFFEE_MQTT_PASSWORD`. Do not edit class names.

## If something is wrong

- Median over 1.00 s: set `day3.infer_conf` to `0.12` and rerun infer. Val score tied 0.05, 0.08, and 0.12. Do not change `imgsz` on the Pi. The file is exported at 640.
- Do not turn `day3.refine` on. It added time and no points.
- Do not turn `autoencoder.fusion_enabled` on. It lowered the class-aware score.
- NCNN will not import: `.venv/bin/pip install -r requirements.txt` again. This folder has no ONNX runtime. Do not spend the window building one.
- One bad image: infer already writes an empty valid result and continues.
