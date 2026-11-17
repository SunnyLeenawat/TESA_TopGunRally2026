"""True-value (TV) calculation and appraisal helpers.

The formula is the organizer's public one:

    TV = P0 * G * M * S

* ``G = (1 - defect_pct / 100) * grade_step``
* ``M`` is the moisture factor (1 below the reference, a linear penalty up to
  the hard limit, then ``hard_factor``).
* ``S`` is the screen-size factor, clamped to the published table.

All constants come from ``GET /v1/calendar -> rules``.  Never hard-code the
numbers here: the competition may publish different values from the sandbox.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def rules_from_calendar(calendar_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``rules`` object from ``GET /v1/calendar``."""
    rules = calendar_payload.get("rules") if isinstance(calendar_payload, Mapping) else None
    if not isinstance(rules, dict):
        raise ValueError(  # noqa: TRY004 - malformed API payload, not a Python argument
            "GET /v1/calendar did not return a rules object"
        )
    required = {"p0", "defect_steps", "moisture", "screen_factors", "bag_kg"}
    missing = required - set(rules)
    if missing:
        raise ValueError(f"calendar rules missing: {sorted(missing)}")
    return rules


def grade(defect_pct: float, rules: Mapping[str, Any]) -> tuple[str, float]:
    """Return ``(grade_name, step_multiplier)`` for a defect percentage.

    Steps are ordered from best to worst.  The first step whose
    ``max_defect_pct`` is not exceeded wins, matching the Excel sheet.
    """
    for step in rules["defect_steps"]:
        if defect_pct <= float(step["max_defect_pct"]):
            return str(step["grade"]), float(step["step"])
    # Defensive fallback: use the last published step.
    last = rules["defect_steps"][-1]
    return str(last["grade"]), float(last["step"])


def moisture_factor(moisture_pct: float, rules: Mapping[str, Any]) -> float:
    """Moisture multiplier ``M``."""
    moisture = rules["moisture"]
    ref = float(moisture["ref"])
    hard_limit = float(moisture["hard_limit"])
    if moisture_pct <= ref:
        return 1.0
    if moisture_pct <= hard_limit:
        return (1 - moisture_pct / 100) / (1 - ref / 100) - float(
            moisture["penalty_per_pp"]
        ) * (moisture_pct - ref)
    return float(moisture["hard_factor"])


def screen_factor(screen: float, rules: Mapping[str, Any]) -> float:
    """Screen multiplier ``S`` with clamping at both ends of the table."""
    table = {int(key): float(value) for key, value in rules["screen_factors"].items()}
    if not table:
        return 1.0
    clamped = min(max(screen, min(table)), max(table))
    return table[int(clamped)]


def true_value(sample: Mapping[str, Any], rules: Mapping[str, Any]) -> float:
    """Compute the estimated true value in THB/kg from one sample.

    Raises ``ValueError`` when a required sample field is missing; the caller
    should skip that lot rather than invent a number.
    """
    try:
        defect_pct = float(sample["defect_pct"])
        moisture_pct = float(sample["moisture_pct"])
        screen = float(sample["screen"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"incomplete sample: {sample!r}") from exc

    _, step = grade(defect_pct, rules)
    grade_part = (1 - defect_pct / 100) * step
    return (
        float(rules["p0"])
        * grade_part
        * moisture_factor(moisture_pct, rules)
        * screen_factor(screen, rules)
    )


def estimate_lots(
    lots: list[Mapping[str, Any]], rules: Mapping[str, Any]
) -> dict[str, float]:
    """Return full-precision TV for every lot that has a usable sample."""
    values: dict[str, float] = {}
    for lot in lots:
        lot_id = lot.get("lot_id")
        sample = lot.get("sample")
        if not lot_id or not isinstance(sample, Mapping):
            continue
        try:
            values[str(lot_id)] = true_value(sample, rules)
        except ValueError:
            continue
    return values


def appraisal_items(
    lots: list[Mapping[str, Any]], rules: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build the ``POST /v1/appraisals`` body (2-decimal estimates)."""
    items: list[dict[str, Any]] = []
    for lot_id, value in estimate_lots(lots, rules).items():
        items.append({"lot_id": lot_id, "est_value_per_kg": round(value, 2)})
    return items
