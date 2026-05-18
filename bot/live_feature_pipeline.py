# -*- coding: utf-8 -*-
"""
V3 即時推論：Binance 1600 根 K 線 → runtime 特徵工程 → 50 維 X_row。

不再使用 combined_features.parquet 末列作為 RF 輸入。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.config import (
    FEATURE_WINDOW,
    FRESHNESS_MAX_SECONDS,
    PREDICTION_REGIME,
    RAW_KLINE_LIMIT,
    RUNTIME_FEATURE_SOURCE,
)
from bot.crypto_market_data import CryptoMarketDataError, get_market_snapshot
from features.runtime_features import build_runtime_cross_market_row, normalize_klines_df
from labels.label_utils import STATE_WINDOW

RV_ROLLING_DAYS = 7
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveSnapshot:
    X_row: pd.DataFrame
    prices: dict[str, float]
    current_rv: dict[str, float]
    historical_avg_vol: dict[str, float]
    feature_timestamp: pd.Timestamp
    latest_kline_time: pd.Timestamp
    kline_delay_seconds: float
    data_delay_seconds: float
    stale: bool
    stale_reason: str
    runtime_feature_source: str
    feature_window: int


def _dailyized_rv_from_klines(df: pd.DataFrame, window: int = STATE_WINDOW) -> float:
    if len(df) < window + 2:
        return float("nan")
    close = df["close"].astype(float)
    log_r = np.log(close / close.shift(1))
    per_min_std = log_r.rolling(window=window, min_periods=window).std().iloc[-1]
    if per_min_std != per_min_std:
        return float("nan")
    return float(per_min_std * np.sqrt(1440))


def build_live_snapshot(*, regime: str = PREDICTION_REGIME) -> LiveSnapshot:
    """組裝 50 維即時特徵列與 Binance 價格／RV／新鮮度。"""
    _ = regime
    now = pd.Timestamp.now(tz="UTC")

    market = get_market_snapshot()
    latest_kline = market.latest_kline_time
    if latest_kline.tzinfo is None:
        latest_kline = latest_kline.tz_localize("UTC")
    else:
        latest_kline = latest_kline.tz_convert("UTC")

    kline_delay = max(0.0, (now - latest_kline).total_seconds())

    prices: dict[str, float] = {}
    current_rv: dict[str, float] = {}
    hist_vol: dict[str, float] = {}

    btc_k = market.klines.get("BTCUSDT")
    eth_k = market.klines.get("ETHUSDT")
    if btc_k is None or btc_k.empty or eth_k is None or eth_k.empty:
        raise ValueError("Binance K 線不足，無法計算 runtime 特徵")

    for sym, k_raw in (("BTCUSDT", btc_k), ("ETHUSDT", eth_k)):
        prices[sym] = float(market.prices.get(sym, float("nan")))
        k = normalize_klines_df(k_raw)
        current_rv[sym] = _dailyized_rv_from_klines(k)
        cut = k.iloc[-RV_ROLLING_DAYS * 1440 :]
        hist_vol[sym] = (
            _dailyized_rv_from_klines(cut) if len(cut) >= STATE_WINDOW else float("nan")
        )

    X_row = build_runtime_cross_market_row(btc_k, eth_k, window=FEATURE_WINDOW)
    last_ts = X_row.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    else:
        last_ts = last_ts.tz_convert("UTC")

    feature_delay = max(0.0, (now - last_ts).total_seconds())
    ts_gap = abs((last_ts - latest_kline).total_seconds())

    stale = (
        kline_delay > FRESHNESS_MAX_SECONDS
        or feature_delay > FRESHNESS_MAX_SECONDS
        or ts_gap > 120
    )
    stale_reason = ""
    if stale:
        parts = []
        if kline_delay > FRESHNESS_MAX_SECONDS:
            parts.append(f"K 線延遲 {kline_delay:.0f}s")
        if feature_delay > FRESHNESS_MAX_SECONDS:
            parts.append(f"特徵延遲 {feature_delay:.0f}s")
        if ts_gap > 120:
            parts.append(f"特徵時間與 K 線相差 {ts_gap:.0f}s")
        stale_reason = "；".join(parts) + f"（>{FRESHNESS_MAX_SECONDS}s）"

    logger.info(
        "V3 live snapshot: source=%s raw_limit=%s window=%s feature_ts=%s kline_ts=%s",
        RUNTIME_FEATURE_SOURCE,
        RAW_KLINE_LIMIT,
        FEATURE_WINDOW,
        last_ts,
        latest_kline,
    )

    return LiveSnapshot(
        X_row=X_row,
        prices=prices,
        current_rv=current_rv,
        historical_avg_vol=hist_vol,
        feature_timestamp=last_ts,
        latest_kline_time=latest_kline,
        kline_delay_seconds=kline_delay,
        data_delay_seconds=feature_delay,
        stale=stale,
        stale_reason=stale_reason,
        runtime_feature_source=RUNTIME_FEATURE_SOURCE,
        feature_window=FEATURE_WINDOW,
    )


def check_market_data_freshness(snap: LiveSnapshot) -> None:
    if not snap.stale:
        return
    raise RuntimeError(snap.stale_reason or "Market data is stale")


def build_live_snapshot_or_raise() -> LiveSnapshot:
    try:
        return build_live_snapshot()
    except CryptoMarketDataError as e:
        raise RuntimeError(
            "即時行情 API 連線失敗，暫停方向性與期權建議。"
        ) from e
