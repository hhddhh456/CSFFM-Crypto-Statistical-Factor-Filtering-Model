# -*- coding: utf-8 -*-
"""Binance 公開 API 即時行情（無 API Key）。"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from bot.config import RAW_KLINE_LIMIT

logger = logging.getLogger(__name__)

_BINANCE_API = "https://api.binance.com/api/v3"
_SYMBOL_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
_REQUEST_TIMEOUT = 15


class CryptoMarketDataError(RuntimeError):
    """即時行情 API 無法取得資料。"""


def _normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper().replace("USDT", "")
    if s in _SYMBOL_MAP:
        return _SYMBOL_MAP[s]
    if s.endswith("USDT"):
        return s
    raise ValueError(f"不支援的 symbol: {symbol}")


def _http_get(path: str, params: dict[str, str | int]) -> object:
    query = urllib.parse.urlencode(params)
    url = f"{_BINANCE_API}{path}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "CSFFM-Bot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise CryptoMarketDataError(f"Binance API 連線失敗: {e}") from e
    except json.JSONDecodeError as e:
        raise CryptoMarketDataError(f"Binance API 回應解析失敗: {e}") from e


def get_latest_price(symbol: Literal["BTC", "ETH"] | str) -> float:
    """取得 USDT 現價。"""
    sym = _normalize_symbol(str(symbol))
    data = _http_get("/ticker/price", {"symbol": sym})
    if not isinstance(data, dict) or "price" not in data:
        raise CryptoMarketDataError(f"無效 ticker 回應: {sym}")
    return float(data["price"])


def _raw_klines_to_rows(raw: list) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in raw:
        if not isinstance(k, list) or len(k) < 7:
            continue
        open_ms = int(k[0])
        close_ms = int(k[6])
        rows.append(
            {
                "open_time": datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc),
            }
        )
    return rows


def get_latest_klines(
    symbol: Literal["BTC", "ETH"] | str,
    *,
    interval: str = "1m",
    limit: int = 1000,
) -> pd.DataFrame:
    """取得最近 K 線；索引為 open_time (UTC)。limit>1000 時自動分頁。"""
    sym = _normalize_symbol(str(symbol))
    need = max(1, int(limit))
    merged: list[list] = []
    end_time_ms: int | None = None

    while len(merged) < need:
        batch_limit = min(1000, need - len(merged))
        params: dict[str, str | int] = {
            "symbol": sym,
            "interval": interval,
            "limit": batch_limit,
        }
        if end_time_ms is not None:
            params["endTime"] = end_time_ms
        raw = _http_get("/klines", params)
        if not isinstance(raw, list) or not raw:
            break
        merged = raw + merged
        end_time_ms = int(raw[0][0]) - 1
        if len(raw) < batch_limit:
            break

    if not merged:
        raise CryptoMarketDataError(f"無 K 線資料: {sym}")

    rows = _raw_klines_to_rows(merged)
    if not rows:
        raise CryptoMarketDataError(f"K 線解析失敗: {sym}")

    df = pd.DataFrame(rows)
    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if len(df) > need:
        df = df.iloc[-need:]
    df.index.name = "open_time"
    return df


@dataclass(frozen=True)
class MarketSnapshot:
    prices: dict[str, float]
    klines: dict[str, pd.DataFrame]
    latest_kline_time: pd.Timestamp
    fetched_at: pd.Timestamp


def get_market_snapshot() -> MarketSnapshot:
    """BTC/ETH 現價與 1m K 線快照。"""
    now = pd.Timestamp.now(tz="UTC")
    prices: dict[str, float] = {}
    klines: dict[str, pd.DataFrame] = {}
    latest = pd.Timestamp.min.tz_localize("UTC")

    for base in ("BTC", "ETH"):
        sym = _SYMBOL_MAP[base]
        try:
            prices[sym] = get_latest_price(base)
            kdf = get_latest_klines(base, interval="1m", limit=RAW_KLINE_LIMIT)
            klines[sym] = kdf
            kt = pd.Timestamp(kdf.index[-1])
            if kt.tzinfo is None:
                kt = kt.tz_localize("UTC")
            else:
                kt = kt.tz_convert("UTC")
            if kt > latest:
                latest = kt
        except Exception as e:
            logger.warning("get_market_snapshot %s failed: %s", sym, e)
            raise CryptoMarketDataError(str(e)) from e

    return MarketSnapshot(
        prices=prices,
        klines=klines,
        latest_kline_time=latest,
        fetched_at=now,
    )
