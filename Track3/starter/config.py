"""Environment settings for the starter agent.

Values are read from ``starter/.env`` first and then from the project-root
``.env``.  The first file that defines a variable wins, so a team can keep the
competition key in ``starter/.env`` without changing the practice-kit config.

The starter accepts both its own names (``MARKET_URL`` / ``TEAM_KEY``) and the
names used by the practice kit (``MARKET_BASE_URL`` / ``MARKET_API_KEY``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

STARTER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STARTER_DIR.parent


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _load_env_files(env_file: str | Path | None) -> None:
    candidates: list[Path] = []
    if env_file:
        candidates.append(Path(env_file))
    candidates.extend([STARTER_DIR / ".env", PROJECT_ROOT / ".env"])
    for path in candidates:
        if path and path.is_file():
            # override=False: the first file loaded keeps precedence.
            load_dotenv(path, override=False)


@dataclass
class Settings:
    market_url: str = "http://127.0.0.1:8000"
    team_key: str = ""

    use_llm: bool = True
    llm_base_url: str = ""
    llm_model: str = "event-llm"
    llm_api_key: str = ""
    llm_timeout: float = 60.0
    llm_thinking: bool = False
    llm_max_tokens: int = 1500

    poll_seconds: float = 2.0
    pace: float = 1.3
    spec_margin_defect: float = 0.15
    spec_margin_moisture: float = 0.10
    spec_margin_screen: float = 0.0
    max_candidates: int = 40

    log_dir: Path = field(default_factory=lambda: STARTER_DIR / "logs")
    dry_run: bool = False

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> Settings:
        _load_env_files(env_file)
        env = os.environ

        market_url = (
            env.get("MARKET_URL")
            or env.get("MARKET_BASE_URL")
            or "http://127.0.0.1:8000"
        ).strip()
        team_key = (env.get("TEAM_KEY") or env.get("MARKET_API_KEY") or "").strip()

        # USE_LLM is the starter switch.  If it is absent, fall back to the
        # practice-kit LLM_PROVIDER: mock means "rules only".
        raw_use_llm = env.get("USE_LLM")
        if raw_use_llm is None:
            use_llm = env.get("LLM_PROVIDER", "organizer").strip().lower() != "mock"
        else:
            use_llm = _as_bool(raw_use_llm, True)

        llm_api_key = (env.get("LLM_API_KEY") or team_key).strip()
        llm_base_url = (env.get("LLM_BASE_URL") or "").strip().rstrip("/")
        llm_model = (env.get("LLM_MODEL") or "event-llm").strip()

        # Without an endpoint/model the agent must still run with rules.
        if use_llm and (not llm_base_url or not llm_model):
            use_llm = False

        log_dir = Path(env.get("LOG_DIR", str(STARTER_DIR / "logs"))).expanduser()
        if not log_dir.is_absolute():
            log_dir = (PROJECT_ROOT / log_dir).resolve()

        return cls(
            market_url=market_url.rstrip("/"),
            team_key=team_key,
            use_llm=use_llm,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_timeout=_as_float(env.get("LLM_TIMEOUT"), 60.0),
            llm_thinking=_as_bool(env.get("LLM_THINKING"), False),
            llm_max_tokens=_as_int(env.get("LLM_MAX_TOKENS"), 1500),
            poll_seconds=min(5.0, max(1.0, _as_float(env.get("POLL_SECONDS"), 2.0))),
            pace=max(0.5, _as_float(env.get("PACE"), 1.3)),
            spec_margin_defect=max(0.0, _as_float(env.get("SPEC_MARGIN_DEFECT"), 0.15)),
            spec_margin_moisture=max(0.0, _as_float(env.get("SPEC_MARGIN_MOISTURE"), 0.10)),
            spec_margin_screen=max(0.0, _as_float(env.get("SPEC_MARGIN_SCREEN"), 0.0)),
            max_candidates=max(5, _as_int(env.get("MAX_CANDIDATES"), 40)),
            log_dir=log_dir,
        )

    @property
    def key_ok(self) -> bool:
        """Reject the placeholder key and keys that are not team keys."""
        key = self.team_key
        return (
            bool(key)
            and key.isascii()
            and key.startswith("sk-T")
            and not key.startswith("sk-T00-")
        )

    @property
    def masked_key(self) -> str:
        if len(self.team_key) <= 10:
            return self.team_key[:4] + "..." if self.team_key else "(empty)"
        return self.team_key[:8] + "..." + self.team_key[-4:]

    def margins(self) -> dict[str, float]:
        return {
            "defect": self.spec_margin_defect,
            "moisture": self.spec_margin_moisture,
            "screen": self.spec_margin_screen,
        }
