# -*- coding: utf-8 -*-
"""
Phase 4 Regime 過濾工具。

時段規則以 **UTC** 為準（與 Data Lake K 線索引一致）。
模組註解附 Asia/Taipei 對照，方便閱讀。

| regime     | UTC 區間        | Taipei 對照（約）   |
|------------|-----------------|---------------------|
| all_day    | 全日            | 全日                |
| asia       | 00:00–06:00     | 08:00–14:00         |
| u_s        | 14:30–21:00     | 22:30–05:00（跨日） |
| settlement | 06:00–08:00     | 14:00–16:00         |
"""

from __future__ import annotations

import pandas as pd

ALL_REGIMES: tuple[str, ...] = ("all_day", "asia", "u_s", "settlement")

REGIME_FOLDER_NAMES: dict[str, str] = {
    "all_day": "all_day",
    "asia": "asia",
    "u_s": "U_S",
    "settlement": "settlement",
}

MIN_REGIME_SAMPLES = 5_000


def regime_folder_name(regime: str) -> str:
    key = regime.strip().lower()
    if key not in REGIME_FOLDER_NAMES:
        raise ValueError(f"未知 regime：{regime}，可用：{list(REGIME_FOLDER_NAMES)}")
    return REGIME_FOLDER_NAMES[key]


def _ensure_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index, utc=True)
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def _regime_mask(index: pd.DatetimeIndex, regime: str) -> pd.Series:
    idx = _ensure_utc_index(index)
    key = regime.strip().lower()

    if key == "all_day":
        return pd.Series(True, index=index)

    hour = idx.hour
    minute = idx.minute
    minutes = hour * 60 + minute

    if key == "asia":
        return (hour >= 0) & (hour < 6)

    if key == "u_s":
        start = 14 * 60 + 30
        end = 21 * 60
        return (minutes >= start) & (minutes < end)

    if key == "settlement":
        return (hour >= 6) & (hour < 8)

    raise ValueError(f"未知 regime：{regime}")


def filter_by_regime(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """
    依 regime 過濾 DataFrame（保留 index 順序）。

    Parameters
    ----------
    df : pd.DataFrame
        索引需為 DatetimeIndex（open_time）。
    regime : str
        all_day | asia | u_s | settlement

    Returns
    -------
    pd.DataFrame
        過濾後子集。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("filter_by_regime 需要 DatetimeIndex 索引")

    mask = _regime_mask(df.index, regime)
    out = df.loc[mask]
    if len(out) < MIN_REGIME_SAMPLES and regime.strip().lower() != "all_day":
        raise ValueError(
            f"regime={regime} 樣本數過少（{len(out)} < {MIN_REGIME_SAMPLES}），"
            "請檢查時段定義或資料範圍"
        )
    return out
