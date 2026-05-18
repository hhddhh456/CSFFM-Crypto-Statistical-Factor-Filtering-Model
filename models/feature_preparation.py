# -*- coding: utf-8 -*-
"""
Phase 4 跨市場特徵合併與資料準備（Multi-timeframe + Regime）。

規則：
- 目標幣種訓練時使用 BTC+ETH 全部特徵（共 50 個：25×2）
- 欄位前綴：btc_ / eth_
- 與目標幣種 labels inner join 對齊
- 可選 regime 過濾（all_day / asia / u_s / settlement）
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from features.feature_utils import FEATURES_PER_SYMBOL
from models.model_utils import Phase4Paths, load_combined_features, load_combined_labels
from models.regime_utils import filter_by_regime


@dataclass(frozen=True)
class PreparedData:
    X: pd.DataFrame
    y_all: pd.DataFrame


def _prefix_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [f"{prefix}{c}" for c in out.columns]
    return out


def prepare_cross_market_features(
    target_symbol: str,
    *,
    regime: str = "all_day",
    paths: Phase4Paths | None = None,
    nan_strategy: Literal["dropna", "ffill_then_dropna"] = "dropna",
    allow_ffill: bool = False,
) -> PreparedData:
    """
    讀取 Phase 2 多時間框架特徵（50 欄），對齊 Phase 3 labels，可選 regime 過濾。

    Args:
        target_symbol: BTCUSDT 或 ETHUSDT
        regime: all_day | asia | u_s | settlement
        nan_strategy: dropna 或 ffill_then_dropna（訓練管線預設禁止 ffill）
        allow_ffill: 僅診斷用；True 時允許 ffill_then_dropna

    Returns:
        PreparedData(X, y_all)
    """
    p4 = paths or Phase4Paths.default()
    tgt = target_symbol.strip().upper()

    if tgt not in ("BTCUSDT", "ETHUSDT"):
        raise ValueError("目前 Phase 4 僅支援 BTCUSDT/ETHUSDT")

    btc_f = load_combined_features("BTCUSDT", paths=p4)
    eth_f = load_combined_features("ETHUSDT", paths=p4)

    expected_per_symbol = FEATURES_PER_SYMBOL
    if len(btc_f.columns) != expected_per_symbol:
        raise ValueError(
            f"BTC 特徵欄數錯誤：預期 {expected_per_symbol}，實際 {len(btc_f.columns)}"
        )
    if len(eth_f.columns) != expected_per_symbol:
        raise ValueError(
            f"ETH 特徵欄數錯誤：預期 {expected_per_symbol}，實際 {len(eth_f.columns)}"
        )

    if nan_strategy == "ffill_then_dropna":
        if not allow_ffill:
            raise ValueError(
                "訓練/回測管線禁止使用 ffill_then_dropna；請使用 dropna。"
                "診斷比較請設 allow_ffill=True。"
            )
        warnings.warn("ffill_then_dropna 僅供診斷，勿用於模型訓練", UserWarning, stacklevel=2)
        btc_f = btc_f.ffill()
        eth_f = eth_f.ffill()
    elif nan_strategy != "dropna":
        raise ValueError("nan_strategy 只能是 dropna 或 ffill_then_dropna")

    X = pd.concat([_prefix_cols(btc_f, "btc_"), _prefix_cols(eth_f, "eth_")], axis=1)
    y_all = load_combined_labels(tgt, paths=p4)

    joined = X.join(y_all, how="inner")
    X2 = joined[X.columns]
    y2 = joined[y_all.columns]

    mask = ~(pd.isna(X2).any(axis=1) | pd.isna(y2).any(axis=1))
    X3 = X2.loc[mask]
    y3 = y2.loc[mask]

    if regime.strip().lower() != "all_day":
        combined = X3.join(y3)
        filtered = filter_by_regime(combined, regime)
        X3 = filtered[X.columns]
        y3 = filtered[y_all.columns]

    expected_total = FEATURES_PER_SYMBOL * 2
    if X3.shape[1] != expected_total:
        raise ValueError(f"跨市場特徵欄數錯誤：預期 {expected_total}，實際 {X3.shape[1]}")

    return PreparedData(X=X3.astype("float64"), y_all=y3)
