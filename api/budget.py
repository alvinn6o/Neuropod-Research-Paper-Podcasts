"""Spend guard.

Per-user daily counters (`rate_limits`) were the only cost control, and they are
not a spend bound: `POST /auth/stub/login` mints a persistent user for any email
with no verification, so an attacker gets a fresh quota per fabricated identity
for free. At Sonnet prices one identity running the cap is roughly $5/day, and a
shell loop can mint thousands.

This module adds the two controls that are actually bounded:

  * a **USD ceiling** read from the `llm_calls` ledger, per day and per month
  * a **global** (not per-user) daily run counter

Both degrade rather than reject. Over budget, `llm_disabled()` routes the
scriptwriter to its zero-cost template and `/ask` to deterministic metadata
answers, so the site still works for a visitor — it just stops spending.

This is defence in depth, not the last line: the real backstop is a spend cap
configured at the provider (Anthropic workspace limits, OpenAI project budgets),
because those cannot be bypassed by a bug in this file.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from . import store_db
from .config import get_settings

logger = logging.getLogger("neuropod.budget")


def budget_state() -> dict:
    """Current spend against the configured caps."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    month_usd = store_db.spend_usd_since(month_start)
    day_usd = store_db.spend_usd_since(day_start)
    runs_today = store_db.global_runs_today()

    reasons = []
    if settings.monthly_budget_usd > 0 and month_usd >= settings.monthly_budget_usd:
        reasons.append(f"monthly budget reached (${month_usd:.2f} >= ${settings.monthly_budget_usd:.2f})")
    if settings.daily_budget_usd > 0 and day_usd >= settings.daily_budget_usd:
        reasons.append(f"daily budget reached (${day_usd:.2f} >= ${settings.daily_budget_usd:.2f})")

    return {
        "month_usd": round(month_usd, 4),
        "day_usd": round(day_usd, 4),
        "monthly_budget_usd": settings.monthly_budget_usd,
        "daily_budget_usd": settings.daily_budget_usd,
        "global_runs_today": runs_today,
        "global_daily_run_limit": settings.global_daily_run_limit,
        "llm_enabled": not reasons,
        "reasons": reasons,
    }


def llm_spend_allowed() -> tuple[bool, str]:
    """(allowed, reason). Reason is empty when allowed."""
    state = budget_state()
    if state["llm_enabled"]:
        return True, ""
    reason = "; ".join(state["reasons"])
    logger.warning("LLM spend blocked: %s", reason)
    return False, reason


def global_capacity_available() -> tuple[bool, str]:
    """Global run counter, across all users.

    Unlike the budget checks this one *rejects*, because a pipeline run that
    cannot call a model has little value — it would just write template scripts
    into the user's feed and look like a broken product.
    """
    settings = get_settings()
    if settings.global_daily_run_limit <= 0:
        return True, ""
    runs = store_db.global_runs_today()
    if runs >= settings.global_daily_run_limit:
        return False, f"global daily run limit reached ({runs}/{settings.global_daily_run_limit})"
    return True, ""
