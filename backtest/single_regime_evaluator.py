# -*- coding: utf-8 -*-
"""
單一 symbol × label × regime 回測評估。
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backtest.evaluation_utils import (
    BacktestPaths,
    DEFAULT_FEE_BPS,
    DEFAULT_SIGNAL_THRESHOLD,
    classification_metrics,
    labels_to_binary,
    load_regime_model,
    load_test_dataset,
    normalize_regime,
    save_regime_outputs,
    simple_strategy_pnl,
)
from backtest.mda_evaluator import compute_backtest_mda
from backtest.visualization import plot_equity_curve


def evaluate_regime(
    symbol: str,
    label_type: str,
    regime: str,
    test_start_date: str = "2024-01-01",
    *,
    fee_bps: float = DEFAULT_FEE_BPS,
    signal_threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    paths: BacktestPaths | None = None,
    save_outputs: bool = True,
) -> dict:
    """
    對指定 regime 模型在對應 regime 時間切片上回測。

    Returns
    -------
    dict
        含 metrics、strategy、top5_mda、輸出路徑等。
    """
    p = paths or BacktestPaths.default()
    regime_n = normalize_regime(regime)

    X, y_all = load_test_dataset(symbol, regime_n, test_start_date)
    if label_type not in y_all.columns:
        raise ValueError(f"缺少標籤欄位 {label_type}")

    y = y_all[label_type].dropna()
    X = X.loc[y.index]

    model = load_regime_model(symbol, label_type, regime_n)
    proba = model.predict_proba(X)[:, 1]
    y_true = labels_to_binary(y)
    y_pred = (proba >= 0.5).astype(np.int8)

    metrics = classification_metrics(y_true, y_pred, proba)
    strat = simple_strategy_pnl(
        y_true, proba, signal_threshold=signal_threshold, fee_bps=fee_bps
    )

    cut = int(len(X) * 0.8)
    X_mda = X.iloc[cut:]
    y_mda = y.iloc[cut:]
    mda_df, top5 = compute_backtest_mda(model, X_mda, y_mda, regime_n)

    pred_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "proba_pos": proba,
            "pnl": strat["pnl_series"],
        },
        index=y.index,
    )

    report = {
        "symbol": symbol.strip().upper(),
        "label_type": label_type,
        "regime": regime_n,
        "test_start_date": test_start_date,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        **metrics,
        "strategy_return": strat["strategy_return"],
        "n_trades": strat["n_trades"],
        "win_rate": strat["win_rate"],
        "signal_threshold": signal_threshold,
        "fee_bps": fee_bps,
        "top5_mda": top5,
    }

    equity_path = None
    if save_outputs:
        save_regime_outputs(p, symbol, regime_n, label_type, report, pred_df, mda_df)
        equity_path = p.label_equity_path(symbol, regime_n, label_type)
        plot_equity_curve(
            pd.Series(strat["pnl_series"], index=y.index),
            equity_path,
            title=f"{symbol} {regime_n} {label_type}",
        )
        report["equity_curve_path"] = str(equity_path)

    return {
        **report,
        "mda": mda_df,
        "predictions": pred_df,
    }


def evaluate_on_regime_slice(
    symbol: str,
    label_type: str,
    model_regime: str,
    data_regime: str,
    test_start_date: str = "2024-01-01",
    *,
    fee_bps: float = DEFAULT_FEE_BPS,
    signal_threshold: float = DEFAULT_SIGNAL_THRESHOLD,
) -> dict:
    """
    用 model_regime 的模型在 data_regime 時間切片上評估（用於 settlement vs all_day 比較）。
    不寫檔，僅回傳指標。
    """
    regime_model = normalize_regime(model_regime)
    regime_data = normalize_regime(data_regime)

    X, y_all = load_test_dataset(symbol, regime_data, test_start_date)
    y = y_all[label_type].dropna()
    X = X.loc[y.index]

    model = load_regime_model(symbol, label_type, regime_model)
    proba = model.predict_proba(X)[:, 1]
    y_true = labels_to_binary(y)
    y_pred = (proba >= 0.5).astype(np.int8)

    metrics = classification_metrics(y_true, y_pred, proba)
    strat = simple_strategy_pnl(
        y_true, proba, signal_threshold=signal_threshold, fee_bps=fee_bps
    )
    return {
        "symbol": symbol,
        "label_type": label_type,
        "model_regime": regime_model,
        "data_regime": regime_data,
        **metrics,
        "strategy_return": strat["strategy_return"],
        "win_rate": strat["win_rate"],
        "n_trades": strat["n_trades"],
    }
