"""Serve rehearsal images and check a posted results.json plus zip.

stdlib HTTP only. Image bytes come from MOCK_IMAGE_DIR. A POST to /submit
must include form parts named results and archive.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import zipfile
from email import message_from_bytes
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from coffee.mock.check import check_tree

logger = logging.getLogger("coffee.mock")

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="python -m coffee.mock.server")
    parser.add_argument("--images", type=Path, default=Path(os.environ.get("MOCK_IMAGE_DIR", "/data/mix")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MOCK_PORT", "8080")))
    parser.add_argument("--report", type=Path, default=Path(os.environ.get("MOCK_REPORT", "/data/outputs/mock_submit/report.json")))
    args = parser.parse_args(argv)
    if not args.images.is_dir():
        logger.error("Image folder not found: %s", args.images)
        return 2
    handler = _handler(args.images.resolve(), args.report)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    logger.info("Mock HTTP on port %s serving %s", args.port, args.images)
    server.serve_forever()
    return 0


def _handler(image_dir: Path, report_path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                _send(self, 200, b'{"ok":true}')
                return
            if self.path == "/images":
                names = sorted(
                    path.name for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
                )
                _send(self, 200, json.dumps(names).encode("utf-8"))
                return
            prefix = "/images/"
            if self.path.startswith(prefix):
                name = Path(self.path[len(prefix):]).name
                source = image_dir / name
                if name != self.path[len(prefix):] or not source.is_file():
                    _send(self, 404, b'{"error":"not found"}')
                    return
                _send(self, 200, source.read_bytes(), "application/octet-stream")
                return
            _send(self, 404, b'{"error":"not found"}')

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/submit":
                _send(self, 404, b'{"error":"not found"}')
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            parts = _parts(body, self.headers.get("Content-Type", ""))
            problems = _accept(parts, report_path)
            status = 200 if not problems else 400
            payload = {"ok": not problems, "problems": problems}
            _send(self, status, json.dumps(payload).encode("utf-8"))
            logger.info("POST /submit ok=%s problems=%s", not problems, len(problems))

        def log_message(self, fmt: str, *args: object) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def _accept(parts: dict[str, tuple[str, bytes]], report_path: Path) -> list[str]:
    if "results" not in parts or "archive" not in parts:
        return ["POST must include form parts 'results' and 'archive'."]
    try:
        posted = json.loads(parts["results"][1].decode("utf-8"))
    except json.JSONDecodeError as exc:
        return [f"results.json is not JSON: {exc}"]
    if not isinstance(posted, list):
        return ["results.json must be a list."]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    (report_path.parent / "results.json").write_bytes(parts["results"][1])
    archive_name = Path(parts["archive"][0]).name or "upload.zip"
    archive_path = report_path.parent / archive_name
    archive_path.write_bytes(parts["archive"][1])
    problems: list[str] = []
    try:
        with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory() as temp:
            _safe_extract(archive, Path(temp))
            problems.extend(check_tree(Path(temp)))
            files = list(Path(temp).rglob("predictions.json"))
            if len(files) == 1:
                inside = json.loads(files[0].read_text(encoding="utf-8"))
                if posted != inside:
                    problems.append("Posted results.json does not match predictions.json inside the zip.")
            else:
                problems.append("The zip does not contain one predictions.json.")
    except zipfile.BadZipFile as exc:
        problems.append(f"archive is not a zip: {exc}")
    names = [str(item.get("image_name")) for item in posted if isinstance(item, dict)]
    report_path.write_text(
        json.dumps({"ok": not problems, "problems": problems, "images": names}, indent=2) + "\n",
        encoding="utf-8",
    )
    return problems


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise zipfile.BadZipFile(f"Zip member escapes the output directory: {member.filename}")
    archive.extractall(destination)


def _parts(body: bytes, content_type: str) -> dict[str, tuple[str, bytes]]:
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = message_from_bytes(header + body, policy=default)
    found: dict[str, tuple[str, bytes]] = {}
    if not message.is_multipart():
        return found
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename() or "upload.bin"
        payload = part.get_payload(decode=True)
        if name and isinstance(payload, bytes):
            found[str(name)] = (filename, payload)
    return found


def _send(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str = "application/json") -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


if __name__ == "__main__":
    raise SystemExit(main())
