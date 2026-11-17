"""The team's brain: turn the market state into a batch of sealed offers.

Everything numeric is computed in Python from ``GET /v1/calendar -> rules``.
The LLM may suggest markups and offers, but :func:`guard_offers` re-checks every
value against the published limits before anything is sent.  When the LLM is
slow, unavailable or returns nonsense the rule planner still produces a complete
legal batch.

Typical tuning points are marked with ``TODO``.  Keep changes small and compare
results across several practice games; one game is mostly noise.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from starter.appraisal import estimate_lots, true_value

# --------------------------------------------------------------------------
# Tuning knobs (the only numbers here; TV constants still come from the API)
# --------------------------------------------------------------------------
# TODO: buyer types do not pay the same.  Start with a mild spread and adjust
# one step at a time using the ACCEPTED / OUTBID / ABOVE_LIMIT feedback.
MARKUP: dict[str, float] = {
    "SPECIALTY": 1.00,
    "COMMERCIAL": 1.00,
    "INSTANT": 1.00,
}

# Hard safety frame.  The LLM and the rules may never leave this range.
LOW, HIGH = 0.85, 1.30

BUYER_TYPES = ("SPECIALTY", "COMMERCIAL", "INSTANT")
DEFAULT_MARGINS = {"defect": 0.15, "moisture": 0.10, "screen": 0.0}


def _bag(rules: Mapping[str, Any]) -> int:
    return max(1, int(rules.get("bag_kg", 60)))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _price(multiplier: float, tv: float) -> float:
    """Clamp a markup to the safe frame and the API's 0-1000 THB/kg range."""
    multiplier = min(max(_number(multiplier, 1.0), LOW), HIGH)
    return max(0.01, round(min(multiplier * tv, 1000.0), 2))


def public_offer(offer: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the four fields accepted by ``POST /v1/offers``."""
    return {
        "rfq_id": str(offer.get("rfq_id", "")),
        "lot_id": str(offer.get("lot_id", "")),
        "kg": int(offer.get("kg", 0)),
        "price_per_kg": round(_number(offer.get("price_per_kg")), 2),
    }


# --------------------------------------------------------------------------
# Spec checks and sales budget
# --------------------------------------------------------------------------
def spec_ok(
    sample: Mapping[str, Any],
    spec: Mapping[str, Any],
    margins: Mapping[str, float] | None = None,
) -> bool:
    """True when the sample passes every buyer limit with a safety margin.

    The buyer checks the *real* lot quality, not the 350 g sample, so a lot
    sitting exactly on a limit can still be rejected.  The default margins are
    deliberately small; raise them if results show ``SPEC_MISMATCH``.
    """
    m = dict(DEFAULT_MARGINS)
    if margins:
        m.update({str(k): _number(v) for k, v in margins.items()})
    try:
        defect = _number(sample["defect_pct"])
        moisture = _number(sample["moisture_pct"])
        screen = _number(sample["screen"])
    except (KeyError, TypeError):
        return False

    max_defect = spec.get("max_defect_pct")
    if max_defect is not None and defect > _number(max_defect) - max(0.0, m["defect"]):
        return False
    max_moisture = spec.get("max_moisture_pct")
    if max_moisture is not None and moisture > _number(max_moisture) - max(
        0.0, m["moisture"]
    ):
        return False
    min_screen = spec.get("min_screen")
    return not (
        min_screen is not None
        and screen < _number(min_screen) + max(0.0, m["screen"])
    )


def _calendar_factor(
    state: Mapping[str, Any], calendar: Mapping[str, Any] | None
) -> float:
    """Look for an explicit demand factor for the current round.

    The organizer may publish the factor under different keys.  We only accept
    a number that is already in the payload; otherwise the factor is 1.0.
    Never invent a festival multiplier from a Thai label.
    """
    if not isinstance(calendar, Mapping):
        return 1.0
    round_info = state.get("round") or {}
    round_id = str(round_info.get("round_id") or "")
    number = round_info.get("number")
    label = str(round_info.get("calendar") or "")

    entries: list[Mapping[str, Any]] = []
    for key in ("rounds", "timeline", "calendar"):
        value = calendar.get(key)
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, Mapping))
        elif isinstance(value, Mapping):
            for name, item in value.items():
                if isinstance(item, Mapping):
                    entries.append({**item, "round_id": item.get("round_id", name)})

    for entry in entries:
        same_round = round_id and str(entry.get("round_id") or "") == round_id
        same_number = number is not None and entry.get("number") == number
        if not (same_round or same_number):
            continue
        for key in (
            "demand_factor",
            "demand_multiplier",
            "sell_multiplier",
            "demand",
            "factor",
            "weight",
        ):
            value = entry.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return min(2.0, max(0.5, float(value)))
        label = str(entry.get("calendar") or label)

    # A label alone is not a number.  Keep the neutral factor and let the team
    # add an explicit rule if the organizer publishes one.
    _ = label
    return 1.0


def sell_budget(
    state: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    calendar: Mapping[str, Any] | None = None,
    pace: float = 1.3,
) -> int:
    """How many kg to offer this round, rounded down to whole bags.

    Spreads the remaining stock over the remaining rounds and allows a small
    overshoot (``pace``) so a good round can sell more.  Festival demand is
    used only when the calendar publishes an explicit numeric factor.
    """
    bag = _bag(rules)
    lots = state.get("lots") or []
    stock = 0
    available = 0
    for lot in lots:
        remaining = max(0, int(lot.get("kg_remaining") or 0))
        avail = max(0, int(lot.get("kg_available") or 0))
        stock += remaining
        available += avail
    if stock <= 0:
        stock = available
    if available <= 0:
        return 0

    capacity = state.get("capacity_left_kg")
    capacity = available if capacity is None else max(0, int(capacity))

    round_info = state.get("round") or {}
    number, of = round_info.get("number"), round_info.get("of")
    try:
        remaining_rounds = max(1, int(of) - int(number) + 1)
    except (TypeError, ValueError):
        remaining_rounds = 1

    target = stock / remaining_rounds * max(0.5, _number(pace, 1.3))
    target *= _calendar_factor(state, calendar)
    budget = min(float(available), float(capacity), target)
    return max(0, int(budget // bag * bag))


# --------------------------------------------------------------------------
# Candidate generation and the rule planner
# --------------------------------------------------------------------------
def _ordered_rfqs(
    rfqs: Sequence[Mapping[str, Any]],
    markup: Mapping[str, float],
    rng: random.Random,
) -> list[Mapping[str, Any]]:
    """Buyers that pay more first; random within a markup band.

    Randomising the order stops every team from chasing the same first RFQ.
    Large RFQs get a higher random key (``u ** (1 / bags)``), so they tend to
    appear earlier without being deterministic.
    """
    decorated: list[tuple[Mapping[str, Any], float, float]] = []
    for rfq in rfqs:
        try:
            bags = max(1, int(rfq.get("bags") or 1))
        except (TypeError, ValueError):
            bags = 1
        key = rng.random() ** (1.0 / bags)
        multiplier = _number(markup.get(str(rfq.get("buyer_type")), 1.0), 1.0)
        decorated.append((rfq, multiplier, key))
    decorated.sort(key=lambda item: (-round(item[1], 3), -item[2]))
    return [rfq for rfq, _, _ in decorated]


def candidate_pairs(
    state: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    markup: Mapping[str, float] | None = None,
    margins: Mapping[str, float] | None = None,
    rng: random.Random | None = None,
    max_pairs: int | None = None,
) -> list[dict[str, Any]]:
    """Every legal (RFQ, lot) pair, in the order the rule planner considers."""
    bag = _bag(rules)
    markup = {**MARKUP, **(markup or {})}
    rng = rng or random.Random()
    tv = estimate_lots(state.get("lots") or [], rules)
    lots = [
        lot
        for lot in (state.get("lots") or [])
        if str(lot.get("lot_id")) in tv and int(lot.get("kg_available") or 0) >= bag
    ]
    rfqs = [
        rfq
        for rfq in (state.get("rfqs") or [])
        if int(rfq.get("kg") or 0) >= bag
    ]

    pairs: list[dict[str, Any]] = []
    for rfq in _ordered_rfqs(rfqs, markup, rng):
        for lot in sorted(lots, key=lambda item: -tv[str(item["lot_id"])]):
            if not spec_ok(lot.get("sample") or {}, rfq.get("spec") or {}, margins):
                continue
            pairs.append(
                {
                    "rfq_id": str(rfq.get("rfq_id")),
                    "lot_id": str(lot.get("lot_id")),
                    "buyer_type": str(rfq.get("buyer_type") or ""),
                    "tv": round(tv[str(lot["lot_id"])], 2),
                    "kg_available": int(lot.get("kg_available") or 0),
                    "rfq_kg": int(rfq.get("kg") or 0),
                }
            )
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs


def rule_plan(
    state: Mapping[str, Any],
    rules: Mapping[str, Any],
    markup: Mapping[str, float] | None = None,
    *,
    margins: Mapping[str, float] | None = None,
    budget: int | None = None,
    seed_offers: Iterable[Mapping[str, Any]] | None = None,
    exclude_rfqs: Iterable[str] | None = None,
    rng: random.Random | None = None,
    max_offers: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministic (but randomised-order) one-offer-per-RFQ plan."""
    bag = _bag(rules)
    markup = {**MARKUP, **(markup or {})}
    rng = rng or random.Random()
    tv = estimate_lots(state.get("lots") or [], rules)
    lots = {
        str(lot["lot_id"]): lot
        for lot in (state.get("lots") or [])
        if str(lot.get("lot_id")) in tv
    }
    left = {
        lot_id: max(0, int(lot.get("kg_available") or 0))
        for lot_id, lot in lots.items()
    }

    room = state.get("capacity_left_kg")
    room = float("inf") if room is None else max(0.0, float(room))
    if budget is not None:
        room = min(room, max(0.0, float(budget)))

    # Offers chosen by the LLM are already part of the batch: subtract them
    # before the rules fill the remaining RFQs.
    for offer in seed_offers or []:
        lot_id = str(offer.get("lot_id", ""))
        kg = max(0, int(offer.get("kg", 0)))
        if lot_id in left:
            left[lot_id] = max(0, left[lot_id] - kg)
        room = max(0.0, room - kg)

    excluded = {str(rfq_id) for rfq_id in (exclude_rfqs or [])}
    rfqs = [
        rfq
        for rfq in (state.get("rfqs") or [])
        if str(rfq.get("rfq_id")) not in excluded
        and int(rfq.get("kg") or 0) >= bag
    ]

    offers: list[dict[str, Any]] = []
    for rfq in _ordered_rfqs(rfqs, markup, rng):
        if room < bag:
            break
        rfq_id = str(rfq.get("rfq_id"))
        best_lot: Mapping[str, Any] | None = None
        for lot in sorted(lots.values(), key=lambda item: -tv[str(item["lot_id"])]):
            lot_id = str(lot["lot_id"])
            if left[lot_id] < bag:
                continue
            if not spec_ok(lot.get("sample") or {}, rfq.get("spec") or {}, margins):
                continue
            best_lot = lot
            break
        if best_lot is None:
            continue

        lot_id = str(best_lot["lot_id"])
        kg = min(left[lot_id], int(rfq.get("kg") or 0), int(room)) // bag * bag
        if kg < bag:
            continue
        multiplier = _number(markup.get(str(rfq.get("buyer_type")), 1.0), 1.0)
        offers.append(
            public_offer(
                {
                    "rfq_id": rfq_id,
                    "lot_id": lot_id,
                    "kg": kg,
                    "price_per_kg": _price(multiplier, tv[lot_id]),
                }
            )
        )
        left[lot_id] -= kg
        room -= kg
        if max_offers is not None and len(offers) >= max_offers:
            break
    return offers


# --------------------------------------------------------------------------
# Guard: the last line of defence before the batch is sent
# --------------------------------------------------------------------------
def guard_offers(
    offers: Iterable[Mapping[str, Any]],
    state: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    margins: Mapping[str, float] | None = None,
    max_total_kg: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Clamp/drop offers so the batch cannot violate a published rule.

    Returns ``(kept, dropped)``.  ``dropped`` items carry a short ``reason`` for
    the decision log.  The function never raises for a malformed offer.
    """
    bag = _bag(rules)
    lots = {
        str(lot.get("lot_id")): lot for lot in (state.get("lots") or [])
    }
    rfqs = {
        str(rfq.get("rfq_id")): rfq for rfq in (state.get("rfqs") or [])
    }
    left = {
        lot_id: max(0, int(lot.get("kg_available") or 0))
        for lot_id, lot in lots.items()
    }
    room = state.get("capacity_left_kg")
    room = float("inf") if room is None else max(0.0, float(room))
    if max_total_kg is not None:
        room = min(room, max(0.0, float(max_total_kg)))

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for raw in offers:
        if not isinstance(raw, Mapping):
            dropped.append({"reason": "offer is not an object"})
            continue
        rfq_id = str(raw.get("rfq_id", ""))
        lot_id = str(raw.get("lot_id", ""))
        reason = ""
        rfq = rfqs.get(rfq_id)
        lot = lots.get(lot_id)
        if rfq is None:
            reason = "unknown rfq_id"
        elif lot is None:
            reason = "unknown lot_id"
        elif (rfq_id, lot_id) in seen:
            reason = "duplicate (rfq, lot)"
        elif not spec_ok(lot.get("sample") or {}, rfq.get("spec") or {}, margins):
            reason = "sample fails buyer spec with margin"
        if reason:
            dropped.append({"rfq_id": rfq_id, "lot_id": lot_id, "reason": reason})
            continue

        try:
            tv = true_value(lot.get("sample") or {}, rules)
        except ValueError:
            dropped.append(
                {"rfq_id": rfq_id, "lot_id": lot_id, "reason": "no usable sample"}
            )
            continue

        kg = max(0, int(_number(raw.get("kg")))) // bag * bag
        kg = min(kg, int(rfq.get("kg") or 0), left[lot_id], int(room))
        if kg < bag:
            dropped.append(
                {
                    "rfq_id": rfq_id,
                    "lot_id": lot_id,
                    "reason": "kg below one bag after clamping",
                }
            )
            continue

        price = _number(raw.get("price_per_kg"), 0.0)
        if price <= 0:
            price = tv
        price = _price(price / tv if tv > 0 else 1.0, tv)

        kept.append(
            public_offer(
                {
                    "rfq_id": rfq_id,
                    "lot_id": lot_id,
                    "kg": kg,
                    "price_per_kg": price,
                }
            )
        )
        seen.add((rfq_id, lot_id))
        left[lot_id] -= kg
        room -= kg

    return kept, dropped


# --------------------------------------------------------------------------
# LLM planner
# --------------------------------------------------------------------------
SYSTEM = """You set prices for a coffee farm in a sealed-offer market.
price = markup x the lot's true value. Buyers remove offers that fail their
spec and offers above a hidden ceiling, then buy the lowest price / true value
first. SPECIALTY usually pays most, INSTANT least. ABOVE_LIMIT means too
expensive for that buyer; OUTBID means another seller was cheaper; ACCEPTED
means sold. Unsold beans are worth only the salvage ratio at the end.
Use only the rfq_id and lot_id values supplied by Python. kg must be a multiple
of bag_kg. Answer only JSON; keep reason to one short sentence."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["markup", "offers", "reason"],
    "properties": {
        "markup": {
            "type": "object",
            "required": list(BUYER_TYPES),
            "properties": {name: {"type": "number"} for name in BUYER_TYPES},
        },
        "offers": {
            "type": "array",
            "maxItems": 40,
            "items": {
                "type": "object",
                "required": ["rfq_id", "lot_id", "kg", "price_per_kg"],
                "properties": {
                    "rfq_id": {"type": "string"},
                    "lot_id": {"type": "string"},
                    "kg": {"type": "integer"},
                    "price_per_kg": {"type": "number"},
                },
            },
        },
        "reason": {"type": "string", "maxLength": 300},
    },
}


def _prompt_payload(
    state: Mapping[str, Any],
    rules: Mapping[str, Any],
    markup: Mapping[str, float],
    memory: Any = None,
    *,
    max_candidates: int = 40,
) -> str:
    tv = estimate_lots(state.get("lots") or [], rules)
    bag = _bag(rules)
    lots = []
    for lot in state.get("lots") or []:
        lot_id = str(lot.get("lot_id"))
        if lot_id not in tv:
            continue
        sample = lot.get("sample") or {}
        lots.append(
            {
                "lot_id": lot_id,
                "tv": round(tv[lot_id], 2),
                "kg_available": int(lot.get("kg_available") or 0),
                "bags": int(lot.get("kg_available") or 0) // bag,
                "sample": {
                    "defect_pct": _number(sample.get("defect_pct")),
                    "moisture_pct": _number(sample.get("moisture_pct")),
                    "screen": _number(sample.get("screen")),
                },
            }
        )
    rfqs = [
        {
            "rfq_id": str(rfq.get("rfq_id")),
            "buyer": str(rfq.get("buyer") or ""),
            "buyer_type": str(rfq.get("buyer_type") or ""),
            "kg": int(rfq.get("kg") or 0),
            "bags": int(rfq.get("bags") or 0),
            "spec": dict(rfq.get("spec") or {}),
        }
        for rfq in (state.get("rfqs") or [])
    ]
    pairs = candidate_pairs(
        state,
        rules,
        markup=markup,
        rng=random.Random(0),
        max_pairs=max_candidates,
    )
    round_info = state.get("round") or {}
    payload = {
        "round": {
            "round_id": round_info.get("round_id"),
            "number": round_info.get("number"),
            "of": round_info.get("of"),
            "calendar": round_info.get("calendar"),
            "kind": round_info.get("kind"),
        },
        "seconds_left": state.get("seconds_left"),
        "bag_kg": bag,
        "capacity_left_kg": state.get("capacity_left_kg"),
        "salvage_ratio": rules.get("salvage_ratio"),
        "current_markup": {key: round(float(value), 3) for key, value in markup.items()},
        "my_recent_results": memory.summary(3) if memory is not None else [],
        "lots": lots,
        "rfqs": rfqs,
        "candidate_pairs": pairs,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def llm_plan(
    state: Mapping[str, Any],
    rules: Mapping[str, Any],
    llm: Any,
    *,
    markup: Mapping[str, float] | None = None,
    memory: Any = None,
    settings: Any = None,
    margins: Mapping[str, float] | None = None,
    budget: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any] | None:
    """Ask the LLM for markups/offers, then let the rules fill the rest."""
    base_markup = {**MARKUP, **(markup or {})}
    seconds_left = state.get("seconds_left")
    timeout = 30.0
    if settings is not None:
        timeout = float(getattr(settings, "llm_timeout", 30.0))
    if seconds_left is not None:
        timeout = min(timeout, max(3.0, float(seconds_left) - 5.0))

    max_candidates = int(getattr(settings, "max_candidates", 40) or 40)
    user = _prompt_payload(
        state, rules, base_markup, memory, max_candidates=max_candidates
    )
    answer = llm.ask_json(SYSTEM, user, SCHEMA, timeout=timeout)
    if not isinstance(answer, Mapping):
        return None

    # ---- markups: keep the old value for anything invalid ----------------
    new_markup = dict(base_markup)
    raw_markup = answer.get("markup")
    if isinstance(raw_markup, Mapping):
        for name in BUYER_TYPES:
            try:
                value = float(raw_markup[name])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                new_markup[name] = round(min(max(value, LOW), HIGH), 3)

    # ---- LLM offers: validate before they are allowed into the batch -----
    raw_offers = answer.get("offers")
    llm_offers: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    if isinstance(raw_offers, list):
        llm_offers, dropped = guard_offers(
            raw_offers,
            state,
            rules,
            margins=margins,
            max_total_kg=budget,
        )

    # ---- rules fill the RFQs the LLM did not cover ------------------------
    rule_offers = rule_plan(
        state,
        rules,
        new_markup,
        margins=margins,
        budget=budget,
        seed_offers=llm_offers,
        exclude_rfqs={offer["rfq_id"] for offer in llm_offers},
        rng=rng,
    )
    combined, extra_dropped = guard_offers(
        [*llm_offers, *rule_offers],
        state,
        rules,
        margins=margins,
        max_total_kg=budget,
    )
    dropped.extend(extra_dropped)

    # ``guard_offers`` may have clamped the LLM's kg; recompute the split from
    # the final batch so the log is honest about what will be sent.
    llm_ids = {
        (offer["rfq_id"], offer["lot_id"])
        for offer in llm_offers
    }
    llm_count = sum(
        1 for offer in combined if (offer["rfq_id"], offer["lot_id"]) in llm_ids
    )
    rule_count = len(combined) - llm_count
    if not combined:
        source = "none"
    elif llm_count and rule_count:
        source = f"llm+rules ({llm_count}+{rule_count})"
    elif llm_count:
        source = "llm"
    else:
        source = "rules"

    notes: list[str] = []
    reason = str(answer.get("reason") or "")[:300]
    if reason:
        notes.append(f"llm: {reason}")
    for item in dropped[:5]:
        notes.append(
            "dropped "
            f"{item.get('rfq_id', '?')}/{item.get('lot_id', '?')}: "
            f"{item.get('reason', '?')}"
        )

    return {
        "offers": combined,
        "source": source,
        "notes": notes,
        "markup": new_markup,
        "reason": reason,
        "budget_kg": int(budget) if budget is not None else None,
        "llm_error": getattr(llm, "last_error", None),
    }


# --------------------------------------------------------------------------
# Public entry point used by main.py
# --------------------------------------------------------------------------
def decide(
    state: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    llm: Any = None,
    settings: Any = None,
    memory: Any = None,
    markup: Mapping[str, float] | None = None,
    margins: Mapping[str, float] | None = None,
    calendar: Mapping[str, Any] | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Return ``{offers, source, notes, markup, ...}`` for one ROUND_OPEN."""
    base_markup = {**MARKUP, **(markup or {})}
    pace = float(getattr(settings, "pace", 1.3) or 1.3)
    if margins is None and settings is not None and hasattr(settings, "margins"):
        margins = settings.margins()
    budget = sell_budget(state, rules, calendar=calendar, pace=pace)
    notes: list[str] = []

    seconds_left = state.get("seconds_left")
    llm_allowed = (
        llm is not None
        and getattr(llm, "enabled", False)
        and (settings is None or bool(getattr(settings, "use_llm", False)))
        and (seconds_left is None or float(seconds_left) >= 8.0)
    )
    if llm_allowed:
        try:
            decision = llm_plan(
                state,
                rules,
                llm,
                markup=base_markup,
                memory=memory,
                settings=settings,
                margins=margins,
                budget=budget,
                rng=rng,
            )
        except Exception as exc:  # noqa: BLE001 - the LLM must never kill a round
            decision = None
            notes.append(f"llm crashed: {type(exc).__name__}: {exc}")
        if decision is not None:
            if notes:
                decision["notes"] = [*notes, *decision.get("notes", [])]
            return decision
        error = getattr(llm, "last_error", None)
        notes.append(f"llm fallback: {error or 'no answer'}")

    offers = rule_plan(
        state,
        rules,
        base_markup,
        margins=margins,
        budget=budget,
        rng=rng,
    )
    return {
        "offers": offers,
        "source": "rules" if offers else "none",
        "notes": notes,
        "markup": base_markup,
        "reason": "",
        "budget_kg": int(budget),
        "llm_error": getattr(llm, "last_error", None) if llm is not None else None,
    }
