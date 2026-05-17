# -*- coding: utf-8 -*-
"""
Roll Measure 與 Roll Impact（市場微觀結構指標）。

本檔案提供：
- compute_roll_measure(df, window): Roll = 2 * sqrt( -Cov(Δp_t, Δp_{t-1}) ) over window
- compute_roll_impact(df, window): RollImpact = Roll / Sum(dollar_volume) over window

公式（忠於文獻標準定義）：
- Δp_t = P_t - P_{t-1}，這裡用 close 價
- Roll_t = 2 × √( max(0, -Cov(Δp_t, Δp_{t-1})) )
  注意：理論上 Cov 可能為負，使 -Cov 為正；若數值上 Cov 為正，則 -Cov 為負 -> 設為 NaN
- RollImpact_t = Roll_t / Σ(dollar_volume_t)（rolling window 內加總）
  dollar_volume_t = close_t * volume_t

NaN 規則：
- 前 W-1 筆一律 NaN（pandas rolling 自然會做到）
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from features.feature_utils import WINDOWS, compute_price_changes


def compute_roll_measure(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Parameters
    ----------
    df : pd.DataFrame
        必須包含 close 欄位（float），index 為 open_time（DatetimeIndex）。
    window : int
        回顧窗口 W（分鐘）。

    Returns
    -------
    pd.Series
        Roll Measure 序列（float64），前 W-1 筆為 NaN。

    Formula
    -------
    Roll = 2 * sqrt( -Cov(Δp_t, Δp_{t-1}) ) over window

    Paper Meaning
    -------------
    Roll Measure 常用來估計「有效買賣價差/交易成本」的 proxy（由 bid-ask bounce 造成的負自相關）。
    """
    if window <= 1:
        raise ValueError("window 必須 >= 2")

    dp = compute_price_changes(df).astype("float64")
    dp_lag = dp.shift(1)

    cov = dp.rolling(window=window).cov(dp_lag)
    x = -cov

    roll = 2.0 * np.sqrt(x.where(x > 0))
    roll.name = f"roll_{window}"
    return roll.astype("float64")


def compute_roll_impact(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Parameters
    ----------
    df : pd.DataFrame
        必須包含 close, volume 欄位（float），index 為 open_time（DatetimeIndex）。
    window : int
        回顧窗口 W（分鐘）。

    Returns
    -------
    pd.Series
        Roll Impact 序列（float64），前 W-1 筆為 NaN。

    Formula
    -------
    RollImpact = Roll / Σ(dollar_volume) over window
    dollar_volume_t = close_t * volume_t

    Paper Meaning
    -------------
    以成交金額規模做標準化，近似衡量在流動性下的交易成本強弱。
    """
    roll = compute_roll_measure(df, window=window)
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("compute_roll_impact 需要 df 具有 close 與 volume 欄位")

    dollar_volume = df["close"].astype("float64") * df["volume"].astype("float64")
    denom = dollar_volume.rolling(window=window).sum()

    impact = roll / denom
    impact.name = f"roll_impact_{window}"
    return impact.astype("float64")


def compute_roll_features(df: pd.DataFrame, windows: Sequence[int] | None = None) -> pd.DataFrame:
    """
    一次計算 Roll Measure 的多窗口特徵。

    支援窗口（預設 WINDOWS）：50, 100, 240, 480, 720 分鐘。
    各欄前 W-1 筆為 NaN。

    Returns
    -------
    pd.DataFrame
        欄位：roll_{W}（每個 W 一欄）；索引與 df 完全對齊。
    """
    ws = tuple(windows) if windows is not None else WINDOWS
    out = pd.DataFrame(index=df.index)
    for w in ws:
        out[f"roll_{w}"] = compute_roll_measure(df, window=w)
    return out.astype("float64")


def compute_roll_impact_features(df: pd.DataFrame, windows: Sequence[int] | None = None) -> pd.DataFrame:
    """
    一次計算 Roll Impact 的多窗口特徵。

    支援窗口（預設 WINDOWS）：50, 100, 240, 480, 720 分鐘。
    各欄前 W-1 筆為 NaN。

    Returns
    -------
    pd.DataFrame
        欄位：roll_impact_{W}（每個 W 一欄）；索引與 df 完全對齊。
    """
    ws = tuple(windows) if windows is not None else WINDOWS
    out = pd.DataFrame(index=df.index)
    for w in ws:
        out[f"roll_impact_{w}"] = compute_roll_impact(df, window=w)
    return out.astype("float64")

