# -*- coding: utf-8 -*-
"""
Realized Volatility Label（已實現增量安全版）。

公式（SOP）：
- Current RV_t = std(log_return) over [t - state_window + 1, t]
- Future  RV_t = std(log_return) over [t + h - state_window + 1, t + h]
- Label = sign(Future RV_t - Current RV_t)，若相等則 -1

實作技巧：
先算 state_series = rolling_std(log_return, state_window)
再 future_state = state_series.shift(-h)
即可得到以 t+h 視窗結束點的未來狀態。
"""

from __future__ import annotations

import pandas as pd

from labels.label_utils import (
    HORIZON_H,
    STATE_WINDOW,
    apply_initial_nan_mask,
    compute_log_returns,
    label_from_states,
    shift_forward,
)


def compute_realized_volatility_label(
    df: pd.DataFrame, state_window: int = STATE_WINDOW, h: int = HORIZON_H
) -> pd.Series:
    """
    Parameters
    ----------
    df : pd.DataFrame
        必須包含 close 欄位；index 為 open_time。
    state_window : int
        狀態窗口（分鐘），固定 1500。
    h : int
        前瞻窗口（分鐘），固定 1500。

    Returns
    -------
    pd.Series
        label_realized_volatility（int8：+1/-1；NaN 表示不可計算區間）

    Notes
    -----
    輸出索引與 raw K 線一致；`label_pipeline` 會再對齊 Phase 2 combined_features 索引。
    """
    r = compute_log_returns(df).astype("float64")
    current = r.rolling(window=state_window, min_periods=state_window).std()
    future = shift_forward(current, h=h)

    label = label_from_states(current, future)
    label = apply_initial_nan_mask(label, state_window=state_window, h=h)
    label.name = "label_realized_volatility"
    return label.astype("Int8")

