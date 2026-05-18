# -*- coding: utf-8 -*-
"""
V3：即時推論特徵工程（與 Phase 2 feature_pipeline 共用同一套 compute_* 公式）。

僅供 Telegram runtime；離線訓練仍使用 Data Lake parquet。
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from features.amihud_measure import compute_amihud_features
from features.feature_utils import FEATURES_PER_SYMBOL, FeaturePaths, WINDOWS
from features.kyle_lambda import compute_kyle_lambda_features
from features.roll_measure import compute_roll_features, compute_roll_impact_features
from features.vpin import compute_vpin_features

logger = logging.getLogger(__name__)

_INDICATOR_COMPUTERS = (
    compute_roll_features,
    compute_roll_impact_features,
    compute_amihud_features,
    compute_kyle_lambda_features,
    compute_vpin_features,
)


def normalize_klines_df(df: pd.DataFrame) -> pd.DataFrame:
    """確保 OHLCV + DatetimeIndex(UTC)。"""
    if df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "open_time" in out.columns:
            out = out.set_index("open_time")
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "open_time"
    required = ("open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"K 線缺少欄位: {missing}")
    return out.sort_index()


def build_symbol_features(
    klines_df: pd.DataFrame,
    *,
    window: int | None = None,
) -> pd.DataFrame:
    """
    單幣 25 維特徵（與 Phase 2 combined_features 欄位名一致，無前綴）。
    在 RAW_KLINE_LIMIT 根 K 線上計算（buffer 供 rolling warm-up）。
    """
    _ = window
    df = normalize_klines_df(klines_df)
    if len(df) < max(WINDOWS) + 2:
        raise ValueError(f"K 線不足：需要至少 {max(WINDOWS) + 2} 根，實際 {len(df)}")

    work = df.copy()
    frames: list[pd.DataFrame] = []
    for fn in _INDICATOR_COMPUTERS:
        frames.append(fn(work))

    combined = pd.concat(frames, axis=1)
    if len(combined.columns) != FEATURES_PER_SYMBOL:
        raise ValueError(
            f"特徵欄數錯誤：預期 {FEATURES_PER_SYMBOL}，實際 {len(combined.columns)}"
        )
    return combined


def _prefix_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [f"{prefix}{c}" for c in out.columns]
    return out


def select_latest_feature_row(
    feature_df: pd.DataFrame,
    *,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    取最新 K 線時間對應列；缺值欄位以同欄向前 ffill（僅 runtime inference）。
    Roll 類指標在 cov>=0 時可能為 NaN，與離線 parquet 全表 dropna 策略對齊。
    """
    if feature_df.empty:
        raise ValueError("特徵 DataFrame 為空")
    cols = required_columns or list(feature_df.columns)
    subset = feature_df[cols] if cols else feature_df
    raw_last = subset.iloc[[-1]]
    filled = subset.ffill()
    row = filled.iloc[[-1]].copy()
    still_nan = [c for c in cols if c in row.columns and row[c].iloc[0] != row[c].iloc[0]]
    if still_nan:
        raise ValueError(f"特徵欄位無法填補（{len(still_nan)}）：{still_nan[:8]}")
    ffilled = [
        c
        for c in cols
        if c in raw_last.columns
        and raw_last[c].iloc[0] != raw_last[c].iloc[0]
        and row[c].iloc[0] == row[c].iloc[0]
    ]
    if ffilled:
        logger.info("runtime inference ffill 欄位（%d）：%s", len(ffilled), ffilled[:8])
    return row


def build_runtime_cross_market_row(
    btc_klines: pd.DataFrame,
    eth_klines: pd.DataFrame,
    *,
    window: int | None = None,
) -> pd.DataFrame:
    """BTC+ETH 前綴合併為單列 50 維 X_row（index = 較晚的末筆 K 線時間）。"""
    btc_feat = build_symbol_features(btc_klines, window=window)
    eth_feat = build_symbol_features(eth_klines, window=window)

    btc_row = select_latest_feature_row(btc_feat)
    eth_row = select_latest_feature_row(eth_feat)

    btc_p = _prefix_cols(btc_row, "btc_")
    eth_p = _prefix_cols(eth_row, "eth_")

    ts = max(btc_row.index[-1], eth_row.index[-1])
    values = {**btc_p.iloc[0].to_dict(), **eth_p.iloc[0].to_dict()}
    combined = pd.DataFrame([values], index=pd.Index([ts], name="open_time"))
    if len(combined.columns) != FEATURES_PER_SYMBOL * 2:
        raise ValueError(
            f"跨市場特徵欄數錯誤：預期 {FEATURES_PER_SYMBOL * 2}，實際 {len(combined.columns)}"
        )
    return combined


@lru_cache(maxsize=1)
def get_canonical_unprefixed_columns(symbol: str = "BTCUSDT") -> tuple[str, ...]:
    """從 Data Lake combined_features 讀取欄位名（僅名稱基準，runtime 不讀末列）。"""
    path = FeaturePaths.default().combined_path(symbol)
    if not path.is_file():
        logger.warning("canonical columns: 找不到 %s", path)
        return tuple()
    return tuple(pd.read_parquet(path).columns)
