"""Cost and quota accounting backed by the ``cost_ledger`` table."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from avd.config import Settings
from avd.db.models import CostLedger
from avd.youtube.client import QuotaExceeded

PACIFIC = ZoneInfo("America/Los_Angeles")  # YouTube Data API quotas reset at midnight PT


class BudgetExceeded(RuntimeError):
    pass


def utc_midnight(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def pacific_midnight(now: datetime | None = None) -> datetime:
    now = (now or datetime.now(UTC)).astimezone(PACIFIC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)


def next_pacific_midnight(now: datetime | None = None) -> datetime:
    return pacific_midnight(now) + timedelta(days=1, minutes=5)


def llm_spent_today(session: Session) -> float:
    total = session.scalar(
        select(func.coalesce(func.sum(CostLedger.cost_usd), 0.0)).where(
            CostLedger.created_at >= utc_midnight(), CostLedger.provider != "youtube_data_api"
        )
    )
    return float(total or 0.0)


def youtube_units_today(session: Session) -> int:
    total = session.scalar(
        select(func.coalesce(func.sum(CostLedger.quota_units), 0)).where(
            CostLedger.created_at >= pacific_midnight(), CostLedger.provider == "youtube_data_api"
        )
    )
    return int(total or 0)


class BudgetManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def remaining(self, session: Session) -> float:
        return max(0.0, self.settings.daily_llm_budget_usd - llm_spent_today(session))

    def remaining_fraction(self, session: Session) -> float:
        if self.settings.daily_llm_budget_usd <= 0:
            return 0.0
        return self.remaining(session) / self.settings.daily_llm_budget_usd

    def can_afford(self, session: Session, estimated_cost: float) -> bool:
        return estimated_cost <= 0 or estimated_cost <= self.remaining(session)


def make_quota_guard(
    session_factory: Callable[[], Session], settings: Settings, job_id: str | None = None
) -> Callable[[str, int], None]:
    """Check-then-record guard for YouTube API quota units (recorded before the call, since
    YouTube charges quota for failed requests as well)."""

    def guard(operation: str, units: int) -> None:
        with session_factory() as s:
            used = youtube_units_today(s)
            limit = settings.youtube_daily_quota_units - settings.youtube_quota_reserve_units
            if used + units > limit:
                raise QuotaExceeded(f"{operation} needs {units} units; {used}/{limit} used today")
            s.add(
                CostLedger(
                    job_id=job_id,
                    provider="youtube_data_api",
                    operation=operation,
                    quota_units=units,
                )
            )
            s.commit()

    return guard
