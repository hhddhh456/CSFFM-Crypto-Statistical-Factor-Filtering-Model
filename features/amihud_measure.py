# -*- coding: utf-8 -*-
"""
Amihud Measure（市場微觀結構指標）。

本檔案提供：
- compute_amihud(df, window): mean( |log_return_t| / dollar_volume_t ) over window

公式（依 SOP）：
- r_t = log(P_t) - log(P_{t-1})
- dollar_volume_t：本專案 Phase 1 規格中 volume 為「USDT 計價成交量」，
  因此這裡的 dollar_volume_t 直接使用 df['volume']（USDT）。
- Amihud_t = mean( |r_t| / dollar_volume_t ) over rolling window

NaN 規則：
- 前 W-1 筆一律 NaN（rolling window 不足）
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from features.feature_utils import WINDOWS, compute_log_returns


def compute_amihud(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Parameters
    ----------
    df : pd.DataFrame
        必須包含 close 與 volume 欄位（volume 為 USDT 計價成交量），index 為 open_time。
    window : int
        回顧窗口 W（分鐘）。

    Returns
    -------
    pd.Series
        Amihud 序列（float64），前 W-1 筆為 NaN。

    Formula
    -------
    Amihud = mean( |log_return_t| / dollar_volume_t ) over window

    Paper Meaning
    -------------
    Amihud 用來衡量「單位成交金額造成的價格變動」，
    值越大通常代表流動性越差（同樣成交量下價格更容易被推動）。
    """
    if window <= 1:
        raise ValueError("window 必須 >= 2")
    if "volume" not in df.columns:
        raise ValueError("compute_amihud 需要 df 具有 volume 欄位（USDT 計價成交量）")

    r = compute_log_returns(df).astype("float64").abs()
    dollar_volume = df["volume"].astype("float64")

    # 避免除以 0：成交量為 0 的分鐘，該分鐘比值視為 NaN（由 rolling mean 忽略）
    ratio = r / dollar_volume.replace(0.0, np.nan)
    amihud = ratio.rolling(window=window).mean()
    amihud.name = f"amihud_{window}"
    return amihud.astype("float64")


def compute_amihud_features(df: pd.DataFrame, windows: Sequence[int] | None = None) -> pd.DataFrame:
    """
    一次計算 Amihud 的多窗口特徵。

    支援窗口（預設 WINDOWS）：50, 100, 240, 480, 720 分鐘。
    各欄前 W-1 筆為 NaN。

    Returns
    -------
    pd.DataFrame
        欄位：amihud_{W}（每個 W 一欄）；索引與 df 完全對齊。
    """
    ws = tuple(windows) if windows is not None else WINDOWS
    out = pd.DataFrame(index=df.index)
    for w in ws:
        out[f"amihud_{w}"] = compute_amihud(df, window=w)
    return out.astype("float64")

