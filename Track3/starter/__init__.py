"""Competition starter agent for the TESA coffee market.

The package follows the organizer API manual in ``hint/คู่มือ-API.md`` and the
starter architecture described in ``hint/สร้าง-agent.ipynb`` / the day-18 slides.

Only ``strategy.py`` should normally be edited by the team.  Everything else is
the transport, validation and logging shell that keeps the agent alive when the
market or the LLM misbehaves.
"""

__all__ = [
    "appraisal",
    "config",
    "llm",
    "main",
    "market",
    "memory",
    "strategy",
]
