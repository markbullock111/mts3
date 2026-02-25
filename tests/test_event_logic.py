from __future__ import annotations

from datetime import datetime, timezone

from inference.event_logic import DailyFirstCheckInDeduper, should_keep_first


def test_should_keep_first_logic():
    t1 = datetime(2026, 2, 25, 8, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 2, 25, 8, 5, tzinfo=timezone.utc)
    assert should_keep_first(None, t1) is True
    assert should_keep_first(t2, t1) is True
    assert should_keep_first(t1, t2) is False


def test_daily_first_checkin_deduper_keeps_earliest_only():
    d = DailyFirstCheckInDeduper()
    t1 = datetime(2026, 2, 25, 8, 10, tzinfo=timezone.utc)
    t0 = datetime(2026, 2, 25, 8, 5, tzinfo=timezone.utc)
    t2 = datetime(2026, 2, 25, 8, 12, tzinfo=timezone.utc)

    r1 = d.consider(101, t1)
    r0 = d.consider(101, t0)
    r2 = d.consider(101, t2)

    assert r1.keep_new is True and r1.reason == "first_seen"
    assert r0.keep_new is True and r0.reason == "earlier_replacement"
    assert r2.keep_new is False and r2.reason == "later_duplicate"
