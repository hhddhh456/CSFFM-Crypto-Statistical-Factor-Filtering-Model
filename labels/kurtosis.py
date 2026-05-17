# -*- coding: utf-8 -*-
"""
Kurtosis Label（峰度標籤，Pearson 峰度）。

公式（SOP）：
- current_state_t = kurtosis(log_return, fisher=False) over state_window
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


def _rolling_kurtosis_pearson(r: pd.Series, window: int) -> pd.Series:
    """
    以 rolling moments 計算 Pearson 峰度（避免 rolling.apply + scipy）。

    kurtosis_pearson = m4 / m2^2
    """
    x = r.astype("float64")
    mu = x.rolling(window=window, min_periods=window).mean()
    ex2 = x.pow(2).rolling(window=window, min_periods=window).mean()
    ex3 = x.pow(3).rolling(window=window, min_periods=window).mean()
    ex4 = x.pow(4).rolling(window=window, min_periods=window).mean()

    m2 = ex2 - mu.pow(2)
    m4 = ex4 - 4 * mu * ex3 + 6 * mu.pow(2) * ex2 - 3 * mu.pow(4)

    k = m4 / m2.pow(2)
    k = k.where(m2 > 0)
    return k


def compute_kurtosis_label(
    df: pd.DataFrame, state_window: int = STATE_WINDOW, h: int = HORIZON_H
) -> pd.Series:
    """
    Returns
    -------
    pd.Series
        label_kurtosis（int8：+1/-1；NaN 表示不可計算區間）

    Notes
    -----
    輸出索引與 raw K 線一致；`label_pipeline` 會再對齊 Phase 2 combined_features 索引。
    """
    r = compute_log_returns(df).astype("float64")
    current = _rolling_kurtosis_pearson(r, state_window)
    future = shift_forward(current, h=h)

    label = label_from_states(current, future)
    label = apply_initial_nan_mask(label, state_window=state_window, h=h)
    label.name = "label_kurtosis"
    return label.astype("Int8")

