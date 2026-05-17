# -*- coding: utf-8 -*-
"""
Skewness Label（偏度標籤）。

公式（SOP）：
- current_state_t = skew(log_return) over state_window
- future_state_t  = current_state_{t+h}
- label = sign(future - current)，若相等則 -1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from labels.label_utils import (
    HORIZON_H,
    STATE_WINDOW,
    apply_initial_nan_mask,
    compute_log_returns,
    label_from_states,
    shift_forward,
)


def _rolling_skewness(r: pd.Series, window: int) -> pd.Series:
    """
    以 rolling moments 計算偏度（避免 rolling.apply + scipy，速度更穩定）。

    skew = m3 / m2^(3/2)
    其中 m_k 為中心動差 E[(x-μ)^k]（以 rolling mean 估計）。
    """
    x = r.astype("float64")
    mu = x.rolling(window=window, min_periods=window).mean()
    m2 = (x.pow(2).rolling(window=window, min_periods=window).mean()) - mu.pow(2)
    m3 = (
        x.pow(3).rolling(window=window, min_periods=window).mean()
        - 3 * mu * (x.pow(2).rolling(window=window, min_periods=window).mean())
        + 2 * mu.pow(3)
    )
    skew = m3 / m2.pow(1.5)
    skew = skew.where(m2 > 0)
    return skew


def compute_skewness_label(
    df: pd.DataFrame, state_window: int = STATE_WINDOW, h: int = HORIZON_H
) -> pd.Series:
    """
    Returns
    -------
    pd.Series
        label_skewness（int8：+1/-1；NaN 表示不可計算區間）

    Notes
    -----
    輸出索引與 raw K 線一致；`label_pipeline` 會再對齊 Phase 2 combined_features 索引。
    """
    r = compute_log_returns(df).astype("float64")
    current = _rolling_skewness(r, state_window)
    future = shift_forward(current, h=h)

    label = label_from_states(current, future)
    label = apply_initial_nan_mask(label, state_window=state_window, h=h)
    label.name = "label_skewness"
    return label.astype("Int8")

