# -*- coding: utf-8 -*-
"""
Kyle’s Lambda（市場微觀結構指標）。

核心概念（依 SOP）：
- 以 rolling window 做 OLS：Δp_t = λ * signed_volume_t + ε_t
- 取迴歸斜率 λ 作為 Kyle's Lambda

定義：
- Δp_t = P_t - P_{t-1}（這裡使用 close）
- signed_volume_t = sign(Δp_t) * volume_t
  注意：本專案 Phase 1 的 volume 為「USDT 計價成交量」，此處仍依 SOP 使用該 volume。

在 rolling window 內，OLS 斜率：
    λ = Cov(Δp, signed_volume) / Var(signed_volume)

NaN 規則：
- 前 W-1 筆一律 NaN
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from features.feature_utils import WINDOWS, compute_price_changes


def compute_kyle_lambda(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Parameters
    ----------
    df : pd.DataFrame
        必須包含 close 與 volume 欄位，index 為 open_time（DatetimeIndex）。
    window : int
        回顧窗口 W（分鐘）。

    Returns
    -------
    pd.Series
        Kyle's Lambda 序列（float64），前 W-1 筆為 NaN。

    Formula
    -------
    OLS: Δp_t = λ * signed_volume_t + ε_t
    signed_volume_t = sign(Δp_t) * volume_t
    λ = Cov(Δp, signed_volume) / Var(signed_volume)  (rolling over window)

    Paper Meaning
    -------------
    λ 近似衡量「單位有向成交量造成的價格衝擊」；
    通常 λ 越大代表市場越不具流動性（同樣交易量更容易推動價格）。
    """
    if window <= 1:
        raise ValueError("window 必須 >= 2")
    if "volume" not in df.columns or "close" not in df.columns:
        raise ValueError("compute_kyle_lambda 需要 df 具有 close 與 volume 欄位")

    dp = compute_price_changes(df).astype("float64")
    vol = df["volume"].astype("float64")

    signed_vol = np.sign(dp) * vol
    signed_vol = pd.Series(signed_vol, index=df.index, name="signed_volume").astype("float64")

    cov = dp.rolling(window=window).cov(signed_vol)
    var = signed_vol.rolling(window=window).var()

    lam = cov / var.replace(0.0, np.nan)
    lam.name = f"kyle_lambda_{window}"
    return lam.astype("float64")


def compute_kyle_lambda_features(df: pd.DataFrame, windows: Sequence[int] | None = None) -> pd.DataFrame:
    """
    一次計算 Kyle's Lambda 的多窗口特徵。

    支援窗口（預設 WINDOWS）：50, 100, 240, 480, 720 分鐘。
    各欄前 W-1 筆為 NaN。

    Returns
    -------
    pd.DataFrame
        欄位：kyle_lambda_{W}（每個 W 一欄）；索引與 df 完全對齊。
    """
    ws = tuple(windows) if windows is not None else WINDOWS
    out = pd.DataFrame(index=df.index)
    for w in ws:
        out[f"kyle_lambda_{w}"] = compute_kyle_lambda(df, window=w)
    return out.astype("float64")

