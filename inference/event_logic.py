from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Iterable

from .geometry import crossed_entry_line


@dataclass
class FinalizedIdentity:
    employee_id: int | None
    method: str  # face|reid|unknown
    confidence: float


@dataclass
class LineCrossConfig:
    roi_polygon: list[list[float]]
    line_p1: tuple[float, float]
    line_p2: tuple[float, float]


@dataclass
class DedupDecision:
    keep_new: bool
    reason: str


class DailyFirstCheckInDeduper:
    """In-memory dedup helper for runtime-side suppression; backend remains source of truth."""

    def __init__(self) -> None:
        self._earliest: dict[tuple[int, str], datetime] = {}

    def consider(self, employee_id: int, ts: datetime) -> DedupDecision:
        day_key = ts.date().isoformat()
        key = (employee_id, day_key)
        existing = self._earliest.get(key)
        if existing is None:
            self._earliest[key] = ts
            return DedupDecision(True, "first_seen")
        if ts < existing:
            self._earliest[key] = ts
            return DedupDecision(True, "earlier_replacement")
        return DedupDecision(False, "later_duplicate")


def should_keep_first(existing_ts: datetime | None, new_ts: datetime) -> bool:
    if existing_ts is None:
        return True
    return new_ts < existing_ts


def is_within_morning_window(ts: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    start = time.fromisoformat(start_hhmm)
    end = time.fromisoformat(end_hhmm)
    t = ts.timetz().replace(tzinfo=None) if ts.tzinfo else ts.time()
    return start <= t <= end
