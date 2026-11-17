#!/bin/sh
# Run a day on the Pi. Docker when the engine is up, otherwise a local venv.
#   ./run.sh 1 /path/to/images /path/to/output
set -eu

DAY="${1:?usage: ./run.sh <1|2|3> <image-or-folder> <output-folder>}"
INPUT="${2:?usage: ./run.sh <1|2|3> <image-or-folder> <output-folder>}"
OUTPUT="${3:?usage: ./run.sh <1|2|3> <image-or-folder> <output-folder>}"
case "$DAY" in
  1|2|3) ;;
  *) echo "Day must be 1, 2, or 3." >&2; exit 2 ;;
esac

ROOT=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
case "$INPUT" in
  /*) ;;
  *) INPUT="$PWD/$INPUT" ;;
esac
if [ ! -e "$INPUT" ]; then
  echo "Input not found: $INPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT"
case "$OUTPUT" in
  /*) ;;
  *) OUTPUT="$PWD/$OUTPUT" ;;
esac

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker build -f "$ROOT/docker/Dockerfile.pi" -t coffee-edge-pi:phase0 "$ROOT"
  exec docker run --rm \
    -e COFFEE_MODEL_DIR=/models \
    -e COFFEE_DATA_DIR=/data \
    -e COFFEE_OUTPUT_DIR=/out \
    -v "$ROOT/models:/models:ro" \
    -v "$INPUT:/input:ro" \
    -v "$OUTPUT:/out" \
    coffee-edge-pi:phase0 \
    python -m coffee.cli --day "$DAY" --input /input --output /out
fi

echo "Docker is not running. Using a local virtualenv." >&2
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -r "$ROOT/requirements/pi.txt"
fi
export COFFEE_MODEL_DIR="$ROOT/models"
export COFFEE_DATA_DIR="$ROOT"
export COFFEE_OUTPUT_DIR="$OUTPUT"
export PYTHONPATH="$ROOT/src"
exec "$ROOT/.venv/bin/python" -m coffee.cli --day "$DAY" --input "$INPUT" --output "$OUTPUT"
