# Coffee-edge. `make` forces the engine socket because the desktop context is absent.
export DOCKER_HOST := unix:///var/run/docker.sock

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# No --project-directory: the compose file lives in docker/, and its build
# context ".." is this repository. --project-directory would resolve ".." too high.
COMPOSE := docker compose --env-file $(ROOT)/.env -f $(ROOT)/docker/docker-compose.yml
COMPOSE_GPU := docker compose --env-file $(ROOT)/.env -f $(ROOT)/docker/docker-compose.yml -f $(ROOT)/docker/docker-compose.gpu.yml

.PHONY: build-train build-pi hello hello-override test test-train \
	mock-data prep synth day3-data day3-train day3-parity day3-infer day3-score day3-mock day3-s5 day3-ae submission train-yolo train-ae export parity ablate \
	infer-day1 infer-day2 infer-day3 test-arm64 pi-bundle

build-pi:
	$(COMPOSE) build pi

build-train:
	$(COMPOSE) build train

hello: build-pi
	$(COMPOSE) run --rm pi python -m coffee.hello

hello-override: build-pi
	$(COMPOSE) run --rm -e COFFEE_DATA_DIR=/tmp/coffee-override pi \
		python -m coffee.hello --day 2 --output /tmp/coffee-out --input /tmp/pile.jpg

PYTHON ?= $(shell if [ -x $(ROOT)/.venv/bin/python ]; then echo $(ROOT)/.venv/bin/python; else echo python3; fi)

test:
	PYTHONPATH=$(ROOT)/src $(PYTHON) -m pytest -q $(ROOT)/tests

test-train: build-train
	$(COMPOSE) run --rm train pytest -q tests

mock-data: build-train
	$(COMPOSE) run --rm train python -m coffee.data mock

prep: build-train
	$(COMPOSE) run --rm train python -m coffee.data prep

synth: build-train
	$(COMPOSE) run --rm train python -m coffee.data synth

day3-data: build-train
	$(COMPOSE) run --rm train python -m coffee.data day3

day3-train: build-train
	$(COMPOSE) run --rm train python -m coffee.train day3

day3-parity: build-train
	$(COMPOSE) run --rm train python -m coffee.parity

day3-infer: build-train
	$(COMPOSE) run --rm train python infer.py --input /data/mix --weights /data/models/yolo26n-seg_ncnn --out /data/outputs/day3_infer

day3-score: build-train
	$(COMPOSE) run --rm train python -m coffee.eval score --pred /data/outputs/day3_infer --labels /data/mix/labels.json --images /data/mix

MOCK := docker compose --env-file $(ROOT)/.env -f $(ROOT)/docker/docker-compose.mock.yml --profile s4

day3-s5: build-train
	$(COMPOSE) run --rm train python -m coffee.train s5
	$(COMPOSE) run --rm train python -m coffee.eval.tune

day3-ae: build-train
	$(COMPOSE) run --rm train python -m coffee.train ae-day3
	$(COMPOSE) run --rm train python -m coffee.eval.ae_ablate

submission: build-train
	set -eu; \
	. $(ROOT)/.env; \
	team=$${COFFEE_TEAM:-team_55}; \
	data=$$HOST_DATA_DIR; \
	test -f "$$data/outputs/runs/day3-s5/weights/best.pt"; \
	test -f "$$data/models/yolo26n-seg_ncnn/model.ncnn.param"; \
	$(COMPOSE) run --rm train python infer.py --input /data/mix --weights /data/models/yolo26n-seg_ncnn --out /data/outputs/day3_submit; \
	dest=$(ROOT)/dist/$${team}_day3; \
	rm -rf "$$dest"; \
	mkdir -p "$$dest/weights" "$$dest/outputs"; \
	cp $(ROOT)/infer.py $(ROOT)/train.py $(ROOT)/client.py $(ROOT)/bench_pi.py "$$dest/"; \
	cp $(ROOT)/requirements/pi.txt "$$dest/requirements.txt"; \
	cp $(ROOT)/RUNBOOK.md "$$dest/README.md"; \
	rsync -a --exclude '__pycache__' --exclude '*.pyc' $(ROOT)/src $(ROOT)/config "$$dest/"; \
	cp "$$data/outputs/runs/day3-s5/weights/best.pt" "$$dest/weights/best.pt"; \
	cp "$$data/models/yolo26n-seg_ncnn/"* "$$dest/weights/"; \
	cp "$$data/outputs/day3_submit/predictions.json" "$$dest/outputs/predictions.json"; \
	cp -a "$$data/outputs/day3_submit/seg_mask" "$$data/outputs/day3_submit/mqtt" "$$data/outputs/day3_submit/vis" "$$dest/outputs/"; \
	venv=/tmp/coffee-submit-venv; \
	if [ ! -x "$$venv/bin/python" ]; then python3 -m venv "$$venv"; fi; \
	"$$venv/bin/pip" install -q -r "$$dest/requirements.txt"; \
	"$$venv/bin/python" -c "import sys; sys.path.insert(0, '$$dest/src'); import coffee.day3_cli; assert 'torch' not in sys.modules; print('no torch')"; \
	"$$venv/bin/python" -c "import cv2, numpy; cv2.imwrite('/tmp/coffee-submit-one.jpg', numpy.full((96, 96, 3), 180, numpy.uint8))"; \
	cd "$$dest" && "$$venv/bin/python" infer.py --input /tmp/coffee-submit-one.jpg --weights "$$dest/weights" --out /tmp/coffee-submit-out; \
	test -f /tmp/coffee-submit-out/predictions.json; \
	rm -f $(ROOT)/dist/$${team}_Day3.zip; \
	cd $(ROOT)/dist && zip -qr $${team}_Day3.zip $${team}_day3; \
	echo "Wrote $(ROOT)/dist/$${team}_Day3.zip"

day3-mock: build-train
	$(MOCK) up -d --wait mosquitto mock-http
	$(MOCK) run --rm client
	$(MOCK) stop

train-yolo: build-train
	$(COMPOSE) run --rm train python -m coffee.train yolo

export: build-train
	$(COMPOSE) run --rm train python -m coffee.train export

parity: build-train
	$(COMPOSE) run --rm train python -m coffee.parity

train-ae: build-train
	$(COMPOSE) run --rm train python -m coffee.train ae

ablate: build-train
	$(COMPOSE) run --rm train python -m coffee.ablate

infer-day1: build-pi
	$(COMPOSE) run --rm pi python -m coffee.cli --day 1 --input /data/mock/test --output /data/outputs/infer/day1

infer-day2: build-pi
	$(COMPOSE) run --rm pi python -m coffee.cli --day 2 --input /data/mock/test --output /data/outputs/infer/day2

infer-day3: build-pi
	$(COMPOSE) run --rm pi python -m coffee.cli --day 3 --input /data/mock/day3/test --output /data/outputs/infer/day3

pi-bundle:
	set -eu; \
	. $(ROOT)/.env; \
	test -d "$$HOST_DATA_DIR/models/yolo26n-seg_ncnn"; \
	rm -rf dist/coffee-edge-pi; \
	mkdir -p dist/coffee-edge-pi/requirements dist/coffee-edge-pi/docker dist/coffee-edge-pi/models; \
	rsync -a --exclude '__pycache__' --exclude '*.pyc' src config dist/coffee-edge-pi/; \
	cp requirements/pi.txt dist/coffee-edge-pi/requirements/pi.txt; \
	cp docker/Dockerfile.pi dist/coffee-edge-pi/docker/Dockerfile.pi; \
	cp docker/pi.dockerignore dist/coffee-edge-pi/.dockerignore; \
	cp infer.py client.py bench_pi.py dist/coffee-edge-pi/; \
	cp scripts/pi-run.sh dist/coffee-edge-pi/run.sh; \
	chmod +x dist/coffee-edge-pi/run.sh dist/coffee-edge-pi/bench_pi.py; \
	cp -a "$$HOST_DATA_DIR/models/yolo26n-seg_ncnn" "$$HOST_DATA_DIR/models/ae_encoder_ncnn" "$$HOST_DATA_DIR/models/ae_bank.npz" dist/coffee-edge-pi/models/; \
	tar -C dist -czf dist/coffee-edge-pi.tar.gz coffee-edge-pi; \
	echo "Wrote dist/coffee-edge-pi.tar.gz from $$HOST_DATA_DIR/models"; \
	echo "On the Pi: tar -xzf coffee-edge-pi.tar.gz && cd coffee-edge-pi && python3 bench_pi.py --input <images> --weights models/yolo26n-seg_ncnn --out <out>"

test-arm64:
	@if ! ls /proc/sys/fs/binfmt_misc 2>/dev/null | grep -q qemu; then \
	  echo "This host has no qemu-aarch64 binfmt, so it cannot run an arm64 container." >&2; \
	  echo "Build on the Pi with ./run.sh. That run is the arm64 test." >&2; \
	  exit 2; \
	fi
	DOCKER_HOST=unix:///var/run/docker.sock docker build --platform linux/arm64 \
	  -f docker/Dockerfile.pi -t coffee-edge-pi:arm64 .
	DOCKER_HOST=unix:///var/run/docker.sock docker run --rm --platform linux/arm64 \
	  coffee-edge-pi:arm64 python -c "import numpy, cv2, ncnn, yaml, requests, paho.mqtt.client; print('arm64 ok')"
