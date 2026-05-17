# -*- coding: utf-8 -*-
"""
即時推論用特徵：combined_features 末筆（Data Lake）+ Binance 公開 API 即時 K 線。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import logging

from bot.config import FRESHNESS_MAX_SECONDS, PREDICTION_REGIME
from bot.crypto_market_data import CryptoMarketDataError, get_market_snapshot
from labels.label_utils import STATE_WINDOW
from models.feature_preparation import prepare_cross_market_features

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


def _dailyized_rv_from_klines(df: pd.DataFrame, window: int = STATE_WINDOW) -> float:
    if len(df) < window + 2:
        return float("nan")
    close = df["close"].astype(float)
    log_r = np.log(close / close.shift(1))
    per_min_std = log_r.rolling(window=window, min_periods=window).std().iloc[-1]
    if per_min_std != per_min_std:
        return float("nan")
    return float(per_min_std * np.sqrt(1440))


def _klines_for_rv(df: pd.DataFrame) -> pd.DataFrame:
    """供 RV 計算：欄位 close、DatetimeIndex。"""
    if df.empty:
        return df
    out = df.copy()
    if "close" not in out.columns:
        raise ValueError("K 線缺少 close 欄位")
    if not isinstance(out.index, pd.DatetimeIndex):
        if "open_time" in out.columns:
            out = out.set_index("open_time")
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out.sort_index()


def build_live_snapshot(*, regime: str = PREDICTION_REGIME) -> LiveSnapshot:
    """組裝 50 維特徵列（Data Lake）與即時 API 價格／RV／新鮮度。"""
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

    for sym in ("BTCUSDT", "ETHUSDT"):
        prices[sym] = float(market.prices.get(sym, float("nan")))
        k_raw = market.klines.get(sym)
        if k_raw is None or k_raw.empty:
            current_rv[sym] = float("nan")
            hist_vol[sym] = float("nan")
            continue
        k = _klines_for_rv(k_raw)
        current_rv[sym] = _dailyized_rv_from_klines(k)
        cut = k.iloc[-RV_ROLLING_DAYS * 1440 :]
        hist_vol[sym] = (
            _dailyized_rv_from_klines(cut) if len(cut) >= STATE_WINDOW else float("nan")
        )

    prepared = prepare_cross_market_features("BTCUSDT", regime="all_day", nan_strategy="dropna")
    X = prepared.X
    if X.empty:
        raise ValueError("無可用特徵列")

    last_ts = X.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    else:
        last_ts = last_ts.tz_convert("UTC")

    feature_delay = max(0.0, (now - last_ts).total_seconds())

    stale = kline_delay > FRESHNESS_MAX_SECONDS
    stale_reason = ""
    if stale:
        stale_reason = f"K 線延遲 {kline_delay:.0f}s（>{FRESHNESS_MAX_SECONDS}s）"

    return LiveSnapshot(
        X_row=X.iloc[[-1]].copy(),
        prices=prices,
        current_rv=current_rv,
        historical_avg_vol=hist_vol,
        feature_timestamp=last_ts,
        latest_kline_time=latest_kline,
        kline_delay_seconds=kline_delay,
        data_delay_seconds=feature_delay,
        stale=stale,
        stale_reason=stale_reason,
    )


def check_market_data_freshness(snap: LiveSnapshot) -> None:
    if not snap.stale:
        return
    reason = snap.stale_reason or "Market data is stale"
    raise RuntimeError(reason)


def build_live_snapshot_or_raise() -> LiveSnapshot:
    """包裝：API 失敗時拋出明確錯誤供 fallback 使用。"""
    try:
        return build_live_snapshot()
    except CryptoMarketDataError as e:
        raise RuntimeError(
            "即時行情 API 連線失敗，暫停方向性與期權建議。"
        ) from e
