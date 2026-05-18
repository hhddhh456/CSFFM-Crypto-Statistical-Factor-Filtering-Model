# -*- coding: utf-8 -*-
"""V3 runtime 特徵工程 sanity tests（合成 K 線，不需網路）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.feature_utils import FEATURES_PER_SYMBOL
from features.runtime_features import (
    build_runtime_cross_market_row,
    build_symbol_features,
    normalize_klines_df,
)


def _synthetic_klines(n: int = 1600, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    log_p = np.cumsum(rng.normal(0, 0.0002, size=n))
    close = 50000.0 * np.exp(log_p)
    vol = rng.uniform(1e5, 5e5, size=n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": vol,
        },
        index=idx,
    )


def test_build_symbol_features_shape():
    k = _synthetic_klines(1600)
    feat = build_symbol_features(k, window=1500)
    assert len(feat.columns) == FEATURES_PER_SYMBOL
    row = feat.iloc[[-1]]
    assert not row.isna().all(axis=None)


def test_build_runtime_cross_market_row_50_columns():
    btc = _synthetic_klines(1600, seed=1)
    eth = _synthetic_klines(1600, seed=2)
    row = build_runtime_cross_market_row(btc, eth, window=1500)
    assert row.shape == (1, FEATURES_PER_SYMBOL * 2)
    assert row.columns[0].startswith("btc_")
    assert any(c.startswith("eth_") for c in row.columns)


def test_normalize_klines_from_columns():
    raw = _synthetic_klines(100).reset_index()
    out = normalize_klines_df(raw)
    assert isinstance(out.index, pd.DatetimeIndex)
