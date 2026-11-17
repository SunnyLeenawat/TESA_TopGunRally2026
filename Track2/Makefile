# Coffee-edge. `make` forces the engine socket because the desktop context is absent.
export DOCKER_HOST := unix:///var/run/docker.sock

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# No --project-directory: the compose file lives in docker/, and its build
# context ".." is this repository. --project-directory would resolve ".." too high.
COMPOSE := docker compose --env-file $(ROOT)/.env -f $(ROOT)/docker/docker-compose.yml
COMPOSE_GPU := docker compose --env-file $(ROOT)/.env -f $(ROOT)/docker/docker-compose.yml -f $(ROOT)/docker/docker-compose.gpu.yml

.PHONY: build-train build-pi hello hello-override test test-train \
	mock-data prep synth train-yolo train-ae export parity ablate \
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

test:
	PYTHONPATH=$(ROOT)/src python3 -m pytest -q $(ROOT)/tests

test-train: build-train
	$(COMPOSE) run --rm train pytest -q tests

mock-data: build-train
	$(COMPOSE) run --rm train python -m coffee.data mock

prep: build-train
	$(COMPOSE) run --rm train python -m coffee.data prep

synth: build-train
	$(COMPOSE) run --rm train python -m coffee.data synth

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
	cp scripts/pi-run.sh dist/coffee-edge-pi/run.sh; \
	chmod +x dist/coffee-edge-pi/run.sh; \
	cp -a "$$HOST_DATA_DIR/models/yolo26n-seg_ncnn" "$$HOST_DATA_DIR/models/ae_encoder_ncnn" "$$HOST_DATA_DIR/models/ae_bank.npz" dist/coffee-edge-pi/models/; \
	tar -C dist -czf dist/coffee-edge-pi.tar.gz coffee-edge-pi; \
	echo "Wrote dist/coffee-edge-pi.tar.gz from $$HOST_DATA_DIR/models"

test-arm64:
	@if ! ls /proc/sys/fs/binfmt_misc 2>/dev/null | grep -q qemu; then \
	  echo "This host has no qemu-aarch64 binfmt, so it cannot run an arm64 container." >&2; \
	  echo "Build on the Pi with ./run.sh. That run is the arm64 test." >&2; \
	  exit 2; \
	fi
	DOCKER_HOST=unix:///var/run/docker.sock docker build --platform linux/arm64 \
	  -f docker/Dockerfile.pi -t coffee-edge-pi:arm64 .
	DOCKER_HOST=unix:///var/run/docker.sock docker run --rm --platform linux/arm64 \
	  coffee-edge-pi:arm64 python -c "import numpy, cv2, ncnn, yaml; print('arm64 ok')"
