"""Per-game memory for the starter agent.

One JSON file per game (``logs/memory-<game>.json``) holds:

* ``tv``      - the appraisal estimates sent for every lot;
* ``markup``  - the price multipliers currently in use;
* ``sent``    - what we offered in each round (with buyer type and markup);
* ``rounds``  - the official results we received for each round.

The file is written atomically so a crash during a round cannot leave a
half-written JSON document behind.  ``summary()`` turns the same data into the
short English lines that the LLM sees in its user prompt, matching the hint
notebook's ``history`` convention.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    """Local ISO timestamp with a timezone offset (human readable in logs)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_name(value: str) -> str:
    keep = "-_.() "
    cleaned = "".join(ch if ch.isalnum() or ch in keep else "-" for ch in str(value))
    return cleaned.strip().replace(" ", "_")[:80] or "game"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class Memory:
    """Mutable per-game state backed by one JSON file."""

    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = Path(path)
        self.data = data

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def new_game(
        cls,
        log_dir: str | Path,
        game_id: str,
        *,
        tv: Mapping[str, float] | None = None,
        lots: int = 0,
        bag_kg: int = 60,
        markup: Mapping[str, float] | None = None,
        market: str = "",
        team_id: str = "",
    ) -> Memory:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "game_id": str(game_id),
            "created": _now(),
            "updated": _now(),
            "market": market,
            "team_id": team_id,
            "lots": int(lots),
            "bag_kg": int(bag_kg),
            "tv": {str(k): round(float(v), 4) for k, v in (tv or {}).items()},
            "markup": {str(k): float(v) for k, v in (markup or {}).items()},
            "appraisal_submitted": False,
            "sent": {},
            "rounds": [],
        }
        memory = cls(log_dir / f"memory-{_safe_name(game_id)}.json", data)
        memory.save()
        return memory

    @classmethod
    def load(cls, path: str | Path) -> Memory:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"{path} does not contain a JSON object")
        data.setdefault("sent", {})
        data.setdefault("rounds", [])
        data.setdefault("markup", {})
        data.setdefault("tv", {})
        return cls(path, data)

    @classmethod
    def load_latest(cls, log_dir: str | Path) -> Memory | None:
        files = sorted(
            Path(log_dir).glob("memory-*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in files:
            try:
                return cls.load(path)
            except (OSError, ValueError):
                continue
        return None

    # ---------------------------------------------------------------- saving
    def save(self) -> None:
        self.data["updated"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    # --------------------------------------------------------------- helpers
    @property
    def game_id(self) -> str:
        return str(self.data.get("game_id", ""))

    @property
    def bag_kg(self) -> int:
        return max(1, _as_int(self.data.get("bag_kg"), 60))

    def tv(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in self.data.get("tv", {}).items()}

    def get_markup(self, default: Mapping[str, float] | None = None) -> dict[str, float]:
        merged = {str(k): float(v) for k, v in (default or {}).items()}
        for key, value in self.data.get("markup", {}).items():
            try:
                merged[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return merged

    def set_markup(self, markup: Mapping[str, float]) -> None:
        self.data["markup"] = {str(k): round(float(v), 4) for k, v in markup.items()}

    def mark_appraisal(self) -> None:
        self.data["appraisal_submitted"] = True

    @property
    def appraisal_submitted(self) -> bool:
        return bool(self.data.get("appraisal_submitted"))

    # --------------------------------------------------------------- records
    def record_sent(
        self,
        round_id: str,
        offers: Iterable[Mapping[str, Any]],
        *,
        markup: Mapping[str, float] | None = None,
        source: str = "",
    ) -> None:
        """Store the offers we submitted this round (idempotent per round)."""
        round_id = str(round_id)
        clean: list[dict[str, Any]] = []
        for offer in offers:
            item = {
                "rfq_id": str(offer.get("rfq_id", "")),
                "lot_id": str(offer.get("lot_id", "")),
                "kg": _as_int(offer.get("kg")),
                "price_per_kg": round(float(offer.get("price_per_kg", 0.0)), 2),
            }
            if offer.get("buyer_type"):
                item["buyer_type"] = str(offer["buyer_type"])
            if offer.get("markup") is not None:
                try:
                    item["markup"] = round(float(offer["markup"]), 4)
                except (TypeError, ValueError):
                    pass
            clean.append(item)

        sent = self.data.setdefault("sent", {})
        sent[round_id] = clean
        if markup is not None:
            self.set_markup(markup)
        else:
            derived: dict[str, float] = {}
            for item in clean:
                if item.get("buyer_type") and item.get("markup") is not None:
                    derived[str(item["buyer_type"])] = float(item["markup"])
            if derived:
                self.set_markup(derived)
        if source:
            self.data.setdefault("sources", {})[round_id] = source
        self.save()

    def record_results(self, result: Mapping[str, Any]) -> bool:
        """Store one round of official results.

        Returns ``True`` when the round was new, ``False`` when it was already
        present (so the caller can log it only once).
        """
        round_id = str(result.get("round_id") or "")
        if not round_id:
            return False

        rounds = self.data.setdefault("rounds", [])
        entry = {
            "round_id": round_id,
            "kind": str(result.get("kind") or ""),
            "my_offers": [dict(o) for o in (result.get("my_offers") or [])],
            "market_summary": [dict(r) for r in (result.get("market_summary") or [])],
            "received": _now(),
        }
        for index, existing in enumerate(rounds):
            if str(existing.get("round_id")) == round_id:
                rounds[index] = entry
                self.save()
                return False
        rounds.append(entry)
        self.save()
        return True

    # --------------------------------------------------------------- summary
    def summary(self, last_n: int = 3) -> list[str]:
        """Short English lines for the LLM prompt (newest last)."""
        bag = self.bag_kg
        lines: list[str] = []
        for round_data in self.data.get("rounds", [])[-max(1, int(last_n)) :]:
            round_id = str(round_data.get("round_id", "?"))
            types = {
                str(r.get("rfq_id")): str(r.get("buyer_type") or "")
                for r in round_data.get("market_summary", [])
            }
            parts: list[str] = []
            for offer in round_data.get("my_offers", []):
                kind = types.get(str(offer.get("rfq_id"))) or str(
                    offer.get("buyer_type") or "?"
                )
                markup = offer.get("markup")
                if markup is None:
                    tv = self.tv().get(str(offer.get("lot_id")))
                    try:
                        markup = (
                            float(offer.get("price_per_kg", 0.0)) / tv if tv else None
                        )
                    except (TypeError, ValueError):
                        markup = None
                multiple = f" x{float(markup):.2f}" if markup else ""
                kg = max(0, _as_int(offer.get("kg")))
                filled = max(0, _as_int(offer.get("filled_kg")))
                parts.append(
                    f"{kind}{multiple} {offer.get('status', '?')} "
                    f"{filled // bag}/{kg // bag} bags"
                )

            market_parts: list[str] = []
            for row in round_data.get("market_summary", []):
                wanted = _as_int(row.get("wanted_kg"))
                bought = _as_int(row.get("bought_kg"))
                if wanted > 0:
                    market_parts.append(
                        f"{row.get('buyer_type') or row.get('rfq_id')} bought "
                        f"{bought / wanted:.0%} of wanted"
                    )

            line = f"{round_id}: my offers [" + ("; ".join(parts) or "none") + "]"
            if market_parts:
                line += "; market: " + "; ".join(market_parts)
            lines.append(line)
        return lines

    def income(self) -> float:
        """Revenue from every recorded result (THB)."""
        total = 0.0
        for round_data in self.data.get("rounds", []):
            for offer in round_data.get("my_offers", []):
                try:
                    total += float(offer.get("filled_kg", 0)) * float(
                        offer.get("price_per_kg", 0)
                    )
                except (TypeError, ValueError):
                    continue
        return total
