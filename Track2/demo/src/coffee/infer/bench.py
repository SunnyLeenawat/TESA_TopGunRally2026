"""Pi 5 bench. The median is the measured pass only. A missing vcgencmd is not a fake zero."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from coffee.config import AppConfig

logger = logging.getLogger("coffee.bench")

CHECKLIST = """\
Pi bench checklist
- Cool the Pi before the timed pass. Idle it if the case is warm.
- vcgencmd get_throttled should print throttled=0x0. Any other value means the clock was held back.
- MemAvailable should stay above 1500 MB on the 8 GB Pi.
- Run this script in the venv, not in Docker. Docker and QEMU times are not the contest number.
- ncnn threads stay at the config value, default 4.
- The first image is a warmup and is not included in the median.
- latency_ms in each mqtt file is that image's own time. The median of those values is what the latency score uses.
"""

_BITS = (
    (1 << 0, "under-voltage now"),
    (1 << 1, "arm frequency capped now"),
    (1 << 2, "throttled now"),
    (1 << 3, "soft temperature limit now"),
    (1 << 16, "under-voltage has occurred"),
    (1 << 17, "arm frequency cap has occurred"),
    (1 << 18, "throttling has occurred"),
    (1 << 19, "soft temperature limit has occurred"),
)


def decode_throttled(value: int) -> list[str]:
    return [label for bit, label in _BITS if value & bit]


def parse_throttled(text: str) -> int | None:
    for token in text.replace("\n", " ").split():
        if token.startswith("throttled="):
            return int(token.split("=", 1)[1], 16)
    return None


def median_ms(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def mem_available_mb() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return None


def read_vcgencmd() -> tuple[str | None, int | None]:
    """Return the raw line and the parsed bitfield. None means this machine has no vcgencmd."""
    try:
        completed = subprocess.run(
            ["vcgencmd", "get_throttled"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.info("vcgencmd is not available: %s", exc)
        return None, None
    text = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        logger.error("vcgencmd get_throttled failed: %s", text)
        return text or None, None
    return text, parse_throttled(text)


def read_temp() -> str | None:
    try:
        completed = subprocess.run(
            ["vcgencmd", "measure_temp"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or "").strip()
    return text or None


def log_platform(cfg: AppConfig) -> None:
    """Print the checklist and the current throttle and memory readings."""
    logger.info("%s", CHECKLIST.rstrip())
    logger.info("ncnn threads=%s infer_conf=%s", cfg.ncnn_threads, cfg.day3_infer_conf)
    available = mem_available_mb()
    if available is None:
        logger.info("MemAvailable could not be read.")
    else:
        logger.info("MemAvailable %s MB", available)
        if available < 1500:
            logger.error("MemAvailable is below 1500 MB. Close other programs before the timed pass.")
    raw, bits = read_vcgencmd()
    if raw is None:
        logger.info("No vcgencmd. This median is not a Raspberry Pi 5 contest measurement.")
        return
    logger.info("Before: %s", raw)
    temp = read_temp()
    if temp:
        logger.info("Before: %s", temp)
    if bits is None:
        return
    flags = decode_throttled(bits)
    if flags:
        logger.error("Throttle flags: %s", ", ".join(flags))
    else:
        logger.info("Throttle flags: none")
