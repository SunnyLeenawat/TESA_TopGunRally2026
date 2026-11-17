"""Competition starter agent: poll the market, appraise, offer, remember.

Run from the project root::

    python -m starter.main --forever

or from inside ``starter/``::

    python main.py --forever

The loop follows the organizer phases (``APPRAISAL`` -> ``ROUND_OPEN`` ->
``ROUND_CLEARING`` -> ``CLOSED``), never submits twice in the same round, and
falls back to the rule planner whenever the LLM is unavailable.  Every decision
is appended to ``logs/decisions.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow both ``python -m starter.main`` and ``python starter/main.py``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter import strategy
from starter.appraisal import appraisal_items, estimate_lots, rules_from_calendar
from starter.config import Settings
from starter.llm import LLMClient
from starter.market import MarketClient, MarketError, MarketUnavailable
from starter.memory import Memory


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class AgentRunner:
    """One market session: state polling, phase handling and logging."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: MarketClient | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or MarketClient(settings.market_url, settings.team_key)
        self.llm = llm if llm is not None else LLMClient(settings)
        self.log_dir = Path(settings.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "decisions.jsonl"
        self.pointer_path = self.log_dir / "current_game.json"

        self.rules: dict[str, Any] | None = None
        self.calendar: Any = None
        self.memory: Memory | None = None
        self.pointer: dict[str, Any] = self._read_json(self.pointer_path) or {}
        self.game_id: str = str(self.pointer.get("game_id") or "")

        self.acted_rounds: set[str] = {
            str(round_id) for round_id in self.pointer.get("acted_rounds", [])
        }
        self.rounds_acted = int(self.pointer.get("rounds_acted") or 0)
        self.seen_results: set[str] = set()
        self.closed_logged = bool(self.pointer.get("finished"))
        self.resume_logged = False
        self.stop = False
        self.max_rounds: int | None = None
        self._reset_attempted = False

        memory_name = self.pointer.get("memory_file")
        if memory_name:
            try:
                self.memory = Memory.load(self.log_dir / str(memory_name))
                self.seen_results = {
                    str(row.get("round_id"))
                    for row in self.memory.data.get("rounds", [])
                    if row.get("round_id")
                }
            except (OSError, ValueError):
                self.memory = None
                self.pointer = {}
                self.game_id = ""

    # --------------------------------------------------------------- logging
    def log(self, event: str, **fields: Any) -> None:
        row = {"time": _now(), "event": event, **_json_safe(fields)}
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            print(f"log write failed: {exc}", file=sys.stderr)

        if event in {
            "new_game",
            "appraisal_sent",
            "offers_sent",
            "market_closed",
            "batch_rejected",
            "offer_rejected",
            "strategy_crashed",
            "round_already_sent",
            "state_error",
        }:
            print(self._console_line(event, row))

    @staticmethod
    def _console_line(event: str, row: Mapping[str, Any]) -> str:
        clock = str(row.get("time", ""))[11:19]
        if event == "offers_sent":
            return (
                f"{clock} offers {row.get('round')} · {row.get('n')} offers "
                f"{row.get('kg')} kg · {row.get('source')} · {row.get('seconds')}s"
            )
        if event == "appraisal_sent":
            return f"{clock} appraisal sent · {row.get('n')} lots"
        if event == "market_closed":
            standings = row.get("standings") or {}
            return (
                f"{clock} market closed · sold {standings.get('kg_sold')} kg · "
                f"{standings.get('revenue_thb')} THB"
            )
        if event == "new_game":
            return f"{clock} new game {row.get('game')} · {row.get('lots')} lots"
        detail = {k: v for k, v in row.items() if k not in ("time", "event")}
        return f"{clock} {event} {json.dumps(detail, ensure_ascii=False)[:200]}"

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _save_pointer(self) -> None:
        self.pointer["updated"] = _now()
        try:
            self.pointer_path.write_text(
                json.dumps(self.pointer, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"pointer write failed: {exc}", file=sys.stderr)

    @staticmethod
    def _lot_signatures(state: Mapping[str, Any]) -> dict[str, str]:
        """Stable per-lot identity: id plus the sample the appraisal used."""
        signatures: dict[str, str] = {}
        for lot in state.get("lots") or []:
            sample = lot.get("sample") or {}
            signatures[str(lot.get("lot_id"))] = "|".join(
                [
                    str(lot.get("kg_total")),
                    str(sample.get("defect_pct")),
                    str(sample.get("moisture_pct")),
                    str(sample.get("screen")),
                ]
            )
        return signatures

    @staticmethod
    def _fingerprint(state: Mapping[str, Any]) -> str:
        signatures = AgentRunner._lot_signatures(state)
        if not signatures:
            return ""
        parts = [f"{lot_id}={signature}" for lot_id, signature in signatures.items()]
        return hashlib.sha1("\n".join(sorted(parts)).encode("utf-8")).hexdigest()

    def _load_calendar(self) -> None:
        payload = self.client.calendar()
        self.rules = rules_from_calendar(payload)
        self.calendar = payload.get("calendar")

    def _touch_pointer(self, state: Mapping[str, Any]) -> None:
        round_info = state.get("round") or {}
        self.pointer.update(
            {
                "game_id": self.game_id,
                "last_round_id": round_info.get("round_id"),
                "last_round_number": round_info.get("number"),
                "acted_rounds": sorted(self.acted_rounds),
                "rounds_acted": self.rounds_acted,
                "finished": bool(self.closed_logged),
            }
        )
        fingerprint = self._fingerprint(state)
        if fingerprint:
            self.pointer["fingerprint"] = fingerprint
        signatures = self._lot_signatures(state)
        if signatures:
            self.pointer["lot_signatures"] = signatures
        self._save_pointer()

    # ------------------------------------------------------------ game state
    def _start_game(self, state: Mapping[str, Any]) -> bool:
        try:
            self._load_calendar()
        except (MarketError, MarketUnavailable, TypeError, ValueError) as exc:
            self.log("state_error", where="calendar", error=str(exc))
            return False

        assert self.rules is not None
        team_id = str(state.get("team_id") or "")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        game_id = f"{stamp}-{team_id or 'team'}"
        tv = estimate_lots(state.get("lots") or [], self.rules)
        self.memory = Memory.new_game(
            self.log_dir,
            game_id,
            tv=tv,
            lots=len(state.get("lots") or []),
            bag_kg=int(self.rules.get("bag_kg", 60)),
            markup=strategy.MARKUP,
            market=str(state.get("market") or ""),
            team_id=team_id,
        )
        self.game_id = game_id
        self.pointer = {
            "game_id": game_id,
            "memory_file": self.memory.path.name,
            "finished": False,
            "last_round_id": (state.get("round") or {}).get("round_id"),
            "last_round_number": (state.get("round") or {}).get("number"),
            "fingerprint": self._fingerprint(state),
            "lot_signatures": self._lot_signatures(state),
            "acted_rounds": [],
            "rounds_acted": 0,
            "created": _now(),
        }
        self.acted_rounds = set()
        self.rounds_acted = 0
        self.seen_results = set()
        self.closed_logged = False
        self.resume_logged = True
        self._reset_attempted = False
        self._save_pointer()
        self.log(
            "new_game",
            game=game_id,
            market=str(state.get("market") or ""),
            team=team_id,
            lots=len(state.get("lots") or []),
            round=(state.get("round") or {}).get("round_id"),
            tv={key: round(value, 2) for key, value in tv.items()},
        )
        return True

    def _ensure_game(self, state: Mapping[str, Any]) -> bool:
        phase = str(state.get("phase") or "")
        lots = state.get("lots") or []
        fingerprint = self._fingerprint(state)
        pointer = self.pointer or {}

        if not pointer:
            if phase in {"SETUP", "APPRAISAL", "ROUND_OPEN"} and lots:
                return self._start_game(state)
            return False

        if pointer.get("finished"):
            if phase in {"SETUP", "APPRAISAL", "ROUND_OPEN"} and lots:
                return self._start_game(state)
            return phase == "CLOSED"

        if self.memory is None:
            name = pointer.get("memory_file")
            if name:
                try:
                    self.memory = Memory.load(self.log_dir / str(name))
                except (OSError, ValueError):
                    self.memory = None
            if self.memory is None:
                return self._start_game(state)

        open_offers = state.get("my_open_offers") or []
        number = (state.get("round") or {}).get("number")
        last_number = pointer.get("last_round_number") or 0
        if not open_offers:
            stored = pointer.get("lot_signatures") or {}
            current = self._lot_signatures(state)
            reset_lots = False
            if stored and current:
                common = set(stored) & set(current)
                sample_changed = any(
                    stored[lot_id] != current[lot_id] for lot_id in common
                )
                # No overlap at all means a completely different lot set.
                # A subset (sold-out lots removed) is still the same game.
                reset_lots = sample_changed or not common
            elif not stored:
                # Backward compatibility with an older pointer file.
                reset_lots = bool(
                    fingerprint
                    and pointer.get("fingerprint")
                    and fingerprint != pointer["fingerprint"]
                    and phase in {"SETUP", "APPRAISAL"}
                )
            round_went_back = isinstance(number, int) and number < int(last_number or 0)
            new_round_one = (
                phase in {"SETUP", "APPRAISAL"}
                and number == 1
                and int(last_number or 0) > 1
            )
            if reset_lots or round_went_back or new_round_one:
                return self._start_game(state)

        if not self.resume_logged:
            self.resume_logged = True
        return True

    # --------------------------------------------------------------- phases
    def handle_state(self, state: Mapping[str, Any]) -> None:
        if state.get("paused"):
            return
        if not self._ensure_game(state):
            return
        if self.rules is None:
            try:
                self._load_calendar()
            except (MarketError, MarketUnavailable, TypeError, ValueError) as exc:
                self.log("state_error", where="calendar", error=str(exc))
                return

        self._handle_results(state)
        phase = str(state.get("phase") or "")
        if phase == "APPRAISAL":
            self._handle_appraisal(state)
        elif phase == "ROUND_OPEN":
            self._handle_round_open(state)
        elif phase == "CLOSED":
            self._handle_closed(state)

    def _handle_results(self, state: Mapping[str, Any]) -> None:
        result = state.get("last_results") or {}
        round_id = str(result.get("round_id") or "")
        if not round_id or round_id in self.seen_results:
            return
        # Ignore a stale result from before this game started (or before the
        # agent joined).  Only rounds we actually acted in belong to this game.
        sent = (
            self.memory.data.get("sent", {}) if self.memory is not None else {}
        )
        if round_id not in self.acted_rounds and round_id not in sent:
            return
        self.seen_results.add(round_id)
        if self.memory is not None:
            self.memory.record_results(result)

    def _handle_appraisal(self, state: Mapping[str, Any]) -> None:
        if state.get("appraisal_submitted"):
            if self.memory is not None and not self.memory.appraisal_submitted:
                self.memory.mark_appraisal()
                self.memory.save()
            return
        if self.memory is not None and self.memory.appraisal_submitted:
            return

        assert self.rules is not None
        items = appraisal_items(state.get("lots") or [], self.rules)
        if not items:
            self.log("appraisal_sent", round=(state.get("round") or {}).get("round_id"), n=0)
            if self.memory is not None:
                self.memory.mark_appraisal()
                self.memory.save()
            return

        if self.settings.dry_run:
            self.log(
                "appraisal_sent",
                round=(state.get("round") or {}).get("round_id"),
                n=len(items),
                items=items,
                dry_run=True,
            )
            if self.memory is not None:
                self.memory.mark_appraisal()
                self.memory.save()
            return

        try:
            self.client.submit_appraisal(items)
        except MarketError as exc:
            if exc.code == "APPRAISAL_ALREADY_SUBMITTED":
                self.log(
                    "appraisal_sent",
                    round=(state.get("round") or {}).get("round_id"),
                    n=len(items),
                    already=True,
                )
            else:
                self.log(
                    "state_error",
                    where="appraisal",
                    code=exc.code,
                    hint=exc.hint,
                    allowed=exc.allowed,
                    message=exc.message,
                )
                return
        except MarketUnavailable as exc:
            self.log("state_error", where="appraisal", error=str(exc))
            return
        else:
            self.log(
                "appraisal_sent",
                round=(state.get("round") or {}).get("round_id"),
                n=len(items),
                items=items,
            )

        if self.memory is not None:
            self.memory.mark_appraisal()
            self.memory.save()

    def _handle_round_open(self, state: Mapping[str, Any]) -> None:
        assert self.rules is not None
        round_info = state.get("round") or {}
        round_id = str(round_info.get("round_id") or "")
        if not round_id:
            return

        open_offers = state.get("my_open_offers") or []
        if open_offers:
            # Restart mid-round: the market already reduced kg_available and
            # capacity_left_kg for these offers.  Do not rebuild a smaller plan.
            if round_id not in self.acted_rounds:
                self.acted_rounds.add(round_id)
                self.rounds_acted += 1
                self.log(
                    "round_already_sent",
                    round=round_id,
                    n=len(open_offers),
                    kg=sum(int(offer.get("kg") or 0) for offer in open_offers),
                    source="existing",
                )
                self._touch_pointer(state)
            return

        if round_id in self.acted_rounds:
            return
        if self.max_rounds is not None and self.rounds_acted >= self.max_rounds:
            self.stop = True
            return

        budget = strategy.sell_budget(
            state, self.rules, calendar=self.calendar, pace=self.settings.pace
        )
        markup = (
            self.memory.get_markup(strategy.MARKUP)
            if self.memory is not None
            else dict(strategy.MARKUP)
        )
        margins = self.settings.margins()

        started = time.monotonic()
        try:
            decision = strategy.decide(
                state,
                self.rules,
                llm=self.llm,
                settings=self.settings,
                memory=self.memory,
                markup=markup,
                margins=margins,
                calendar=self.calendar,
            )
        except Exception as exc:  # noqa: BLE001 - a bad strategy must not kill the agent
            self.log(
                "strategy_crashed",
                round=round_id,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=6),
            )
            try:
                decision = strategy.decide(
                    state,
                    self.rules,
                    llm=None,
                    settings=self.settings,
                    memory=self.memory,
                    markup=markup,
                    margins=margins,
                    calendar=self.calendar,
                )
            except Exception as fallback_exc:  # noqa: BLE001 - rules fallback must be reported
                self.log(
                    "state_error",
                    where="strategy",
                    round=round_id,
                    error=f"{type(fallback_exc).__name__}: {fallback_exc}",
                )
                return
        seconds = round(time.monotonic() - started, 1)

        offers, dropped = strategy.guard_offers(
            decision.get("offers") or [],
            state,
            self.rules,
            margins=margins,
            max_total_kg=budget,
        )
        for item in dropped:
            self.log(
                "offer_rejected",
                round=round_id,
                stage="guard",
                rfq_id=item.get("rfq_id"),
                lot_id=item.get("lot_id"),
                reason=item.get("reason"),
            )

        source = str(decision.get("source") or ("rules" if offers else "none"))
        notes = list(decision.get("notes") or [])
        if not offers:
            self.log(
                "offers_sent",
                round=round_id,
                source="none",
                n=0,
                kg=0,
                seconds=seconds,
                offers=[],
                notes=notes,
                budget_kg=budget,
            )
            self.acted_rounds.add(round_id)
            self.rounds_acted += 1
            self._touch_pointer(state)
            return

        enriched = self._enrich(offers, state)
        if self.settings.dry_run:
            self.log(
                "offers_sent",
                round=round_id,
                source=source,
                n=len(enriched),
                kg=sum(offer["kg"] for offer in enriched),
                seconds=seconds,
                offers=enriched,
                notes=notes,
                budget_kg=budget,
                dry_run=True,
            )
            self.acted_rounds.add(round_id)
            self.rounds_acted += 1
            self._touch_pointer(state)
            return

        fallback = False
        try:
            response = self.client.submit_offers(offers, replace_existing=True)
        except MarketError as exc:
            if exc.code == "BATCH_REJECTED":
                self.log(
                    "batch_rejected",
                    round=round_id,
                    n=len(offers),
                    allowed=exc.allowed,
                    message=exc.message,
                )
                accepted = self._submit_individual(offers, round_id)
                if not accepted:
                    self.acted_rounds.add(round_id)
                    self.rounds_acted += 1
                    self._touch_pointer(state)
                    return
                enriched = self._enrich(accepted, state)
                fallback = True
            elif exc.code in {"WRONG_PHASE", "ROUND_CLOSED"}:
                self.log(
                    "offer_rejected",
                    round=round_id,
                    stage="batch",
                    code=exc.code,
                    hint=exc.hint,
                    message=exc.message,
                )
                self.acted_rounds.add(round_id)
                self.rounds_acted += 1
                self._touch_pointer(state)
                return
            elif exc.code == "RATE_LIMITED":
                self.log(
                    "state_error",
                    where="offers",
                    round=round_id,
                    code=exc.code,
                    hint=exc.hint,
                    allowed=exc.allowed,
                    message=exc.message,
                )
                return
            else:
                self.log(
                    "offer_rejected",
                    round=round_id,
                    stage="batch",
                    code=exc.code,
                    hint=exc.hint,
                    allowed=exc.allowed,
                    message=exc.message,
                )
                if 400 <= int(exc.status or 0) < 500:
                    self.acted_rounds.add(round_id)
                    self.rounds_acted += 1
                    self._touch_pointer(state)
                return
        except MarketUnavailable as exc:
            self.log("state_error", where="offers", round=round_id, error=str(exc))
            return
        else:
            accepted = self._merge_response(offers, response)
            enriched = self._enrich(accepted, state)

        if self.memory is not None:
            self.memory.record_sent(
                round_id,
                enriched,
                markup=decision.get("markup") or markup,
                source=source,
            )
        self.log(
            "offers_sent",
            round=round_id,
            source=source,
            n=len(enriched),
            kg=sum(offer["kg"] for offer in enriched),
            seconds=seconds,
            offers=enriched,
            notes=notes,
            budget_kg=budget,
            fallback=fallback,
        )
        self.acted_rounds.add(round_id)
        self.rounds_acted += 1
        self._touch_pointer(state)
        if self.max_rounds is not None and self.rounds_acted >= self.max_rounds:
            self.stop = True

    def _handle_closed(self, state: Mapping[str, Any]) -> None:
        if self.closed_logged:
            return
        self.closed_logged = True
        self.log(
            "market_closed",
            game=self.game_id,
            standings=state.get("standings") or {},
            income=round(self.memory.income(), 2) if self.memory is not None else None,
        )
        self.pointer["finished"] = True
        self._save_pointer()

    # --------------------------------------------------------------- helpers
    def _enrich(
        self, offers: list[Mapping[str, Any]], state: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        assert self.rules is not None
        types = {
            str(rfq.get("rfq_id")): str(rfq.get("buyer_type") or "")
            for rfq in (state.get("rfqs") or [])
        }
        tv = estimate_lots(state.get("lots") or [], self.rules)
        enriched: list[dict[str, Any]] = []
        for offer in offers:
            item = {
                "rfq_id": str(offer.get("rfq_id", "")),
                "lot_id": str(offer.get("lot_id", "")),
                "kg": int(offer.get("kg", 0)),
                "price_per_kg": round(float(offer.get("price_per_kg", 0.0)), 2),
                "buyer_type": types.get(str(offer.get("rfq_id", "")), ""),
            }
            if offer.get("offer_id"):
                item["offer_id"] = str(offer["offer_id"])
            lot_tv = tv.get(str(offer.get("lot_id", "")))
            if lot_tv:
                item["markup"] = round(item["price_per_kg"] / lot_tv, 4)
            enriched.append(item)
        return enriched

    @staticmethod
    def _merge_response(
        offers: list[Mapping[str, Any]], response: Any
    ) -> list[dict[str, Any]]:
        merged = [dict(offer) for offer in offers]
        if not isinstance(response, Mapping):
            return merged
        returned = response.get("offers")
        if not isinstance(returned, list):
            return merged
        by_key = {
            (str(item.get("rfq_id")), str(item.get("lot_id"))): item
            for item in returned
            if isinstance(item, Mapping)
        }
        for offer in merged:
            key = (str(offer.get("rfq_id")), str(offer.get("lot_id")))
            match = by_key.get(key)
            if match and match.get("offer_id"):
                offer["offer_id"] = str(match["offer_id"])
        return merged

    def _submit_individual(
        self, offers: list[Mapping[str, Any]], round_id: str
    ) -> list[dict[str, Any]]:
        accepted: list[dict[str, Any]] = []
        for offer in offers:
            public = strategy.public_offer(offer)
            try:
                response = self.client.submit_offer(public)
            except MarketError as exc:
                self.log(
                    "offer_rejected",
                    round=round_id,
                    stage="individual",
                    code=exc.code,
                    hint=exc.hint,
                    allowed=exc.allowed,
                    message=exc.message,
                    offer=public,
                )
                if exc.code == "RATE_LIMITED" and exc.retry_after:
                    time.sleep(min(5.0, max(0.0, exc.retry_after)))
                continue
            except MarketUnavailable as exc:
                self.log(
                    "state_error", where="offers", round=round_id, error=str(exc)
                )
                break
            item = dict(offer)
            if isinstance(response, Mapping) and response.get("offer_id"):
                item["offer_id"] = str(response["offer_id"])
            accepted.append(item)
        return accepted

    # ------------------------------------------------------------------- run
    def _sandbox_reset(self) -> None:
        try:
            self.client.reset()
        except MarketError as exc:
            if exc.code != "NOT_AVAILABLE":
                self.log(
                    "state_error",
                    where="reset",
                    code=exc.code,
                    hint=exc.hint,
                    message=exc.message,
                )
        except MarketUnavailable as exc:
            self.log("state_error", where="reset", error=str(exc))
        else:
            self.pointer = {}
            self.memory = None
            self.game_id = ""
            self.acted_rounds = set()
            self.rounds_acted = 0
            self.seen_results = set()
            self.closed_logged = False
            self.resume_logged = False
            self._reset_attempted = False

    def run(
        self,
        *,
        forever: bool = False,
        max_rounds: int | None = None,
        reset: bool = False,
    ) -> None:
        self.max_rounds = max_rounds
        if reset:
            self._sandbox_reset()

        while not self.stop:
            try:
                state = self.client.state()
            except MarketUnavailable as exc:
                self.log("state_error", where="state", error=str(exc))
                time.sleep(max(1.0, self.settings.poll_seconds))
                continue
            except MarketError as exc:
                self.log(
                    "state_error",
                    where="state",
                    code=exc.code,
                    hint=exc.hint,
                    allowed=exc.allowed,
                    message=exc.message,
                )
                if exc.code == "UNAUTHORIZED":
                    raise SystemExit(2)
                time.sleep(max(1.0, self.settings.poll_seconds))
                continue

            try:
                self.handle_state(state)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - keep polling after any handler bug
                self.log(
                    "state_error",
                    where="handle",
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(limit=6),
                )

            if self.stop:
                break
            if (
                not forever
                and str(state.get("phase")) == "CLOSED"
                and self.closed_logged
            ):
                break
            if self.max_rounds is not None and self.rounds_acted >= self.max_rounds:
                break

            if (
                forever
                and str(state.get("phase")) == "CLOSED"
                and str(state.get("market")) == "sandbox"
                and not self._reset_attempted
            ):
                self._reset_attempted = True
                self._sandbox_reset()

            time.sleep(max(1.0, min(5.0, self.settings.poll_seconds)))

    def close(self) -> None:
        self.client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TESA coffee market starter agent (organizer API v1)."
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="keep polling after a game closes; sandbox games are reset automatically",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="stop after this many ROUND_OPEN rounds (default: until the market closes)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="sandbox only: reset the current game before starting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and log decisions but never POST appraisals or offers",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="optional .env path (default: starter/.env then project .env)",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=None,
        help="seconds between GET /v1/state calls (1-5, default 2)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env(args.env_file)
    if args.poll is not None:
        settings.poll_seconds = max(1.0, min(5.0, float(args.poll)))
    settings.dry_run = bool(args.dry_run)

    if not settings.key_ok:
        print(
            "TEAM_KEY / MARKET_API_KEY is missing, not a team key, or still the "
            "placeholder (sk-T00-...). Put the key from the organizer in "
            "starter/.env or the project .env.",
            file=sys.stderr,
        )
        return 2

    llm_state = "on" if (settings.use_llm and settings.llm_base_url) else "off"
    print(
        f"market={settings.market_url} key={settings.masked_key} "
        f"llm={llm_state} dry_run={settings.dry_run} "
        f"logs={settings.log_dir}"
    )

    runner = AgentRunner(settings)
    try:
        runner.run(
            forever=args.forever,
            max_rounds=args.max_rounds,
            reset=args.reset,
        )
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
