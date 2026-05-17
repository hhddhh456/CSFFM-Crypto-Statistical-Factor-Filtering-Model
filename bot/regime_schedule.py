# -*- coding: utf-8 -*-
"""排程時段與 regime 對應；美股開盤夏令時 fallback。"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from bot.config import SCHEDULE_REGIME_MAP
from models.regime_utils import _regime_mask


def resolve_regime_for_slot(slot: str, *, at: datetime | None = None) -> str:
    """
    slot: asia_pre | settlement_pre | us_open | us_close
    """
    base = SCHEDULE_REGIME_MAP.get(slot, "all_day")
    if slot not in ("us_open", "us_close"):
        return base
    ts = at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    idx = pd.DatetimeIndex([pd.Timestamp(ts)])
    if bool(_regime_mask(idx, "u_s")[0]):
        return "u_s"
    return "all_day"


def regime_for_utc_now(*, at: datetime | None = None) -> str:
    """手動 /predict：依當前 UTC 選最特定 regime。"""
    ts = at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    idx = pd.DatetimeIndex([pd.Timestamp(ts)])
    for reg in ("settlement", "asia", "u_s"):
        if bool(_regime_mask(idx, reg)[0]):
            return reg
    return "all_day"


def format_times_for_report(*, at: datetime | None = None) -> dict[str, str]:
    ts = at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    utc = pd.Timestamp(ts).tz_convert("UTC")
    tw = utc.tz_convert(ZoneInfo("Asia/Taipei"))
    ny = utc.tz_convert(ZoneInfo("America/New_York"))
    return {
        "taiwan_time": tw.strftime("%Y-%m-%d %H:%M"),
        "utc_time": utc.strftime("%Y-%m-%d %H:%M"),
        "ny_time": ny.strftime("%Y-%m-%d %H:%M"),
    }
