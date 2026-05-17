# -*- coding: utf-8 -*-
"""
VPIN（Volume-synchronized Probability of Informed Trading）

本專案採用 SOP 指定的 Bulk Volume Classification（BVC）版本：

1) 先計算價格變化與其波動度
   - Δp_t = P_t - P_{t-1}（使用 close）
   - σ_t = rolling_std(Δp_t) over window

2) BVC 買方成交量（比例）
   - BVC_buy_t = volume_t * Φ( Δp_t / σ_t )
     其中 Φ 為標準常態 CDF
   - BVC_sell_t = volume_t - BVC_buy_t

3) VPIN
   - VPIN_t = | Σ(BVC_buy) - Σ(BVC_sell) | / Σ(volume)  over window

化簡：
   Σ(BVC_buy) - Σ(BVC_sell) = 2*Σ(BVC_buy) - Σ(volume)

NaN 規則：
- 前 W-1 筆為 NaN（rolling window 不足）
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from features.feature_utils import WINDOWS, compute_price_changes


def compute_vpin(df: pd.DataFrame, window: int = 50) -> pd.Series:
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
        VPIN 序列（float64），前 W-1 筆為 NaN。

    Formula
    -------
    BVC_buy_t  = volume_t * Φ(Δp_t / σ_t)
    BVC_sell_t = volume_t - BVC_buy_t
    VPIN_t = |Σ(BVC_buy) - Σ(BVC_sell)| / Σ(volume)  over window

    Paper Meaning
    -------------
    VPIN 常被視為「資訊交易風險」或「買賣不平衡程度」的 proxy，
    越高通常代表市場更可能處於不穩定或流動性惡化狀態。
    """
    if window <= 1:
        raise ValueError("window 必須 >= 2")
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("compute_vpin 需要 df 具有 close 與 volume 欄位")

    dp = compute_price_changes(df).astype("float64")
    vol = df["volume"].astype("float64")

    # σ_t：用 Δp 的 rolling 標準差（min_periods=window 確保前 W-1 為 NaN）
    sigma = dp.rolling(window=window, min_periods=window).std()

    z = dp / sigma.replace(0.0, np.nan)
    phi = pd.Series(norm.cdf(z), index=df.index).astype("float64")

    bvc_buy = vol * phi
    # 以 rolling sum 計算 VPIN
    sum_buy = bvc_buy.rolling(window=window, min_periods=window).sum()
    sum_vol = vol.rolling(window=window, min_periods=window).sum()

    imbalance = (2.0 * sum_buy - sum_vol).abs()
    vpin = imbalance / sum_vol.replace(0.0, np.nan)
    vpin.name = f"vpin_{window}"
    return vpin.astype("float64")


def compute_vpin_features(df: pd.DataFrame, windows: Sequence[int] | None = None) -> pd.DataFrame:
    """
    一次計算 VPIN 的多窗口特徵。

    支援窗口（預設 WINDOWS）：50, 100, 240, 480, 720 分鐘。
    各欄前 W-1 筆為 NaN。

    Returns
    -------
    pd.DataFrame
        欄位：vpin_{W}（每個 W 一欄）；索引與 df 完全對齊。
    """
    ws = tuple(windows) if windows is not None else WINDOWS
    out = pd.DataFrame(index=df.index)
    for w in ws:
        out[f"vpin_{w}"] = compute_vpin(df, window=w)
    return out.astype("float64")

