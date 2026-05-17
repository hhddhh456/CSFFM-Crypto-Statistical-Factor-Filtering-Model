# -*- coding: utf-8 -*-
"""
Sequential Correlation（rolling autocorrelation, lag=1）Label。

公式（SOP）：
- current_state_t = rolling_autocorr(log_return, lag=1) over state_window
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


def _autocorr_lag1(x: np.ndarray) -> float:
    """
    計算序列 x 的 lag=1 自相關。
    若變異為 0 或長度不足，回傳 NaN。
    """
    if x.size < 2:
        return np.nan
    x0 = x[:-1]
    x1 = x[1:]
    if np.std(x0) == 0 or np.std(x1) == 0:
        return np.nan
    return float(np.corrcoef(x0, x1)[0, 1])


def compute_sequential_correlation_label(
    df: pd.DataFrame, state_window: int = STATE_WINDOW, h: int = HORIZON_H
) -> pd.Series:
    """
    Returns
    -------
    pd.Series
        label_sequential_correlation（int8：+1/-1；NaN 表示不可計算區間）

    Notes
    -----
    輸出索引與 raw K 線一致；`label_pipeline` 會再對齊 Phase 2 combined_features 索引。
    """
    r = compute_log_returns(df).astype("float64")
    current = r.rolling(window=state_window, min_periods=state_window).apply(
        _autocorr_lag1, raw=True
    )
    future = shift_forward(current, h=h)

    label = label_from_states(current, future)
    label = apply_initial_nan_mask(label, state_window=state_window, h=h)
    label.name = "label_sequential_correlation"
    return label.astype("Int8")

