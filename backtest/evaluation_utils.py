# -*- coding: utf-8 -*-
"""
Phase 5 回測共用工具：路徑、載入模型/資料、分類指標、簡易策略 PnL。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from models.feature_preparation import PreparedData, prepare_cross_market_features
from models.model_utils import Phase4Paths, load_combined_features, load_combined_labels, load_model
from models.regime_utils import regime_folder_name

DEFAULT_SIGNAL_THRESHOLD = 0.55
DEFAULT_FEE_BPS = 4.0

LABEL_ORDER = [
    "label_realized_volatility",
    "label_sequential_correlation",
    "label_skewness",
    "label_kurtosis",
    "label_jarque_bera",
]


def normalize_regime(regime: str) -> str:
    key = regime.strip().lower()
    if key in ("europe_us", "europe-us", "us"):
        return "u_s"
    return key


@dataclass(frozen=True)
class BacktestPaths:
    root: Path

    @staticmethod
    def default(project_root: Optional[Path] = None) -> "BacktestPaths":
        pr = (project_root or Path(__file__).resolve().parents[1]).resolve()
        return BacktestPaths(root=pr / "backtest")

    def symbol_regime_dir(self, symbol: str, regime: str) -> Path:
        sym = symbol.strip().upper()
        folder = regime_folder_name(normalize_regime(regime))
        return self.root / sym / folder

    def label_report_path(self, symbol: str, regime: str, label_type: str) -> Path:
        return self.symbol_regime_dir(symbol, regime) / f"report_{label_type}.json"

    def label_predictions_path(self, symbol: str, regime: str, label_type: str) -> Path:
        return self.symbol_regime_dir(symbol, regime) / f"predictions_{label_type}.parquet"

    def label_mda_path(self, symbol: str, regime: str, label_type: str) -> Path:
        return self.symbol_regime_dir(symbol, regime) / f"mda_{label_type}.csv"

    def label_equity_path(self, symbol: str, regime: str, label_type: str) -> Path:
        return self.symbol_regime_dir(symbol, regime) / f"equity_curve_{label_type}.png"

    def regime_comparison_path(self) -> Path:
        return self.root / "regime_comparison.csv"

    def settlement_focus_path(self) -> Path:
        return self.root / "settlement_focus_report.md"

    def final_report_path(self) -> Path:
        return self.root / "final_evaluation_report.md"

    def diagnostics_root(self) -> Path:
        return self.root / "diagnostics"

    def diagnostic_symbol_regime_dir(self, symbol: str, regime: str) -> Path:
        sym = symbol.strip().upper()
        folder = regime_folder_name(normalize_regime(regime))
        return self.diagnostics_root() / sym / folder

    def oos_result_path(self, symbol: str, regime: str, label_type: str) -> Path:
        return self.diagnostic_symbol_regime_dir(symbol, regime) / f"oos_{label_type}.json"

    def leakage_report_path(self) -> Path:
        return self.root / "leakage_diagnosis_report.md"

    def regime_oos_comparison_path(self) -> Path:
        return self.root / "regime_oos_comparison.md"

    def oos_comparison_path(self) -> Path:
        return self.root / "oos_comparison.csv"

    def final_diagnosis_path(self) -> Path:
        return self.root / "final_diagnosis_report.md"

    def oos_evaluation_summary_path(self) -> Path:
        return self.root / "oos_evaluation_summary.md"

    def window_mda_summary_path(self) -> Path:
        return self.root / "window_mda_summary.csv"

    def window_oos_comparison_path(self) -> Path:
        return self.root / "window_oos_comparison.csv"

    def window_analysis_summary_path(self) -> Path:
        return self.root / "window_analysis_summary.md"


def load_regime_model(symbol: str, label_type: str, regime: str):
    return load_model(symbol, label_type, regime=normalize_regime(regime))


def _utc_timestamp(date_str: str, *, end_of_day: bool = False) -> pd.Timestamp:
    ts = pd.Timestamp(date_str, tz="UTC")
    if end_of_day:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def load_prepared_dataset(symbol: str, regime: str) -> PreparedData:
    """載入已對齊、regime 過濾後的 X 與 y_all。"""
    return prepare_cross_market_features(symbol, regime=normalize_regime(regime))


def split_xy_by_label(
    prepared: PreparedData,
    label_type: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if label_type not in prepared.y_all.columns:
        raise ValueError(f"缺少標籤欄位 {label_type}")
    y = prepared.y_all[label_type].dropna()
    X = prepared.X.loc[y.index]
    return X, y


def load_dataset_date_range(
    symbol: str,
    regime: str,
    label_type: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    依 UTC 日期區間截取 X, y（含 end 當日全天）。
    start/end 為 None 表示不限制該側。
    """
    X, y = split_xy_by_label(load_prepared_dataset(symbol, regime), label_type)
    if start is not None:
        X = X.loc[X.index >= _utc_timestamp(start)]
        y = y.loc[y.index >= _utc_timestamp(start)]
    if end is not None:
        end_ts = _utc_timestamp(end, end_of_day=True)
        X = X.loc[X.index <= end_ts]
        y = y.loc[y.index <= end_ts]
    if len(X) == 0:
        raise ValueError(f"{symbol} {regime} {label_type} 在指定日期區間無樣本")
    return X, y


def train_oos_masks(
    index: pd.DatetimeIndex,
    *,
    train_end: str,
    test_start: str,
) -> tuple[pd.Series, pd.Series]:
    """train_mask: index <= train_end；oos_mask: index >= test_start。"""
    train_end_ts = _utc_timestamp(train_end, end_of_day=True)
    test_start_ts = _utc_timestamp(test_start)
    train_mask = index <= train_end_ts
    oos_mask = index >= test_start_ts
    return train_mask, oos_mask


def load_test_dataset(
    symbol: str,
    regime: str,
    test_start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    載入測試用 X, y_all（已 regime 過濾 + dropna），並截取 test_start_date 之後。
    """
    regime_n = normalize_regime(regime)
    prepared = prepare_cross_market_features(symbol, regime=regime_n)
    start = pd.Timestamp(test_start_date, tz="UTC")
    mask = prepared.X.index >= start
    X = prepared.X.loc[mask]
    y_all = prepared.y_all.loc[mask]
    if len(X) == 0:
        raise ValueError(f"{symbol} {regime_n} 在 {test_start_date} 之後無測試樣本")
    return X, y_all


def labels_to_binary(y: pd.Series) -> np.ndarray:
    """+1/-1 label → 1/0 for sklearn。"""
    return (y.astype("int8") == 1).astype(np.int8).to_numpy()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    out: dict = {"n_samples": int(len(y_true))}
    if len(np.unique(y_true)) < 2:
        out.update(
            {
                "auc": float("nan"),
                "pr_auc": float("nan"),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float("nan"),
            }
        )
        return out
    out["auc"] = float(roc_auc_score(y_true, y_proba))
    out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return out


def simple_strategy_pnl(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    signal_threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> dict:
    """
    簡易策略：|proba-0.5| 夠大時視為有訊號；預測正確 +1、錯誤 -1，並扣雙邊手續費。
  回傳逐步 pnl 序列與摘要統計。
    """
    y_pred = (y_proba >= 0.5).astype(np.int8)
    confidence = np.abs(y_proba - 0.5) + 0.5
    trade_mask = confidence >= signal_threshold
    fee = fee_bps / 10_000.0

    pnl = np.zeros(len(y_true), dtype=np.float64)
    for i in range(len(y_true)):
        if not trade_mask[i]:
            continue
        correct = y_pred[i] == y_true[i]
        pnl[i] = (1.0 if correct else -1.0) - 2.0 * fee

    traded = pnl[trade_mask]
    win_rate = float((traded > 0).mean()) if len(traded) else 0.0
    return {
        "pnl_series": pnl,
        "strategy_return": float(pnl.sum()),
        "n_trades": int(trade_mask.sum()),
        "win_rate": win_rate,
        "signal_threshold": signal_threshold,
        "fee_bps": fee_bps,
    }


def save_regime_outputs(
    paths: BacktestPaths,
    symbol: str,
    regime: str,
    label_type: str,
    report: dict,
    predictions: pd.DataFrame,
    mda_df: pd.DataFrame,
) -> None:
    out_dir = paths.symbol_regime_dir(symbol, regime)
    out_dir.mkdir(parents=True, exist_ok=True)

    rp = paths.label_report_path(symbol, regime, label_type)
    with rp.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    predictions.to_parquet(paths.label_predictions_path(symbol, regime, label_type), index=True)
    mda_df.to_csv(paths.label_mda_path(symbol, regime, label_type), index=False, encoding="utf-8-sig")
