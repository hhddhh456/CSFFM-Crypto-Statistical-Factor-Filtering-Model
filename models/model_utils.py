# -*- coding: utf-8 -*-
"""
Phase 4 共用工具（Model Utils）。

你已選擇的介面：
- 輸入：Data_Lake processed
  - features：C:\\Data_Lake\\processed\\crypto_microstructure\\features\\{symbol}\\combined_features.parquet
  - labels：C:\\Data_Lake\\processed\\crypto_microstructure\\labels\\{symbol}\\combined_labels.parquet
- 輸出：專案內 models/

時間序列分割：
- Walk-forward 多折評估
- purge = 1500 分鐘（避免 lookahead leakage）
"""
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import joblib
import numpy as np
import pandas as pd

from utils.config import get_data_lake_root


# Phase 4 參數（固定）
PURGE_MINUTES = 1500
RANDOM_STATE = 42
# 訓練資料截止（UTC，含當日全天）；2025+ 留作 OOS 評估
DEFAULT_TRAIN_END_DATE = "2024-12-31"


def train_end_timestamp(train_end_date: str) -> pd.Timestamp:
    """train_end_date 當日 23:59:59.999999 UTC。"""
    ts = pd.Timestamp(train_end_date, tz="UTC")
    return ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def mask_index_up_to_train_end(index: pd.DatetimeIndex, train_end_date: str) -> pd.Series:
    end_ts = train_end_timestamp(train_end_date)
    return index <= end_ts


@dataclass(frozen=True)
class Phase4Paths:
    """集中管理 Phase 4 的讀寫路徑。"""

    data_lake_root: Path
    project_root: Path

    @staticmethod
    def default(project_root: Optional[Path] = None) -> "Phase4Paths":
        pr = (project_root or Path(__file__).resolve().parents[1]).resolve()
        return Phase4Paths(data_lake_root=get_data_lake_root(), project_root=pr)

    def features_combined_path(self, symbol: str) -> Path:
        sym = symbol.strip().upper()
        return (
            self.data_lake_root
            / "processed"
            / "crypto_microstructure"
            / "features"
            / sym
            / "combined_features.parquet"
        )

    def labels_combined_path(self, symbol: str) -> Path:
        sym = symbol.strip().upper()
        return (
            self.data_lake_root
            / "processed"
            / "crypto_microstructure"
            / "labels"
            / sym
            / "combined_labels.parquet"
        )

    def models_root(self) -> Path:
        return self.project_root / "models"

    def symbol_models_dir(self, symbol: str) -> Path:
        sym = symbol.strip().upper()
        return self.models_root() / sym

    def symbol_regime_dir(self, symbol: str, regime: str) -> Path:
        from models.regime_utils import regime_folder_name

        return self.symbol_models_dir(symbol) / regime_folder_name(regime)

    def model_path(self, symbol: str, regime: str, label_type: str) -> Path:
        return self.symbol_regime_dir(symbol, regime) / f"model_{label_type}.pkl"

    def regime_report_path(self, symbol: str, regime: str) -> Path:
        return self.symbol_regime_dir(symbol, regime) / "training_report.json"

    def regime_mda_path(self, symbol: str, regime: str) -> Path:
        return self.symbol_regime_dir(symbol, regime) / "feature_importance_mda.csv"

    def comparison_summary_path(self) -> Path:
        return self.models_root() / "model_comparison_summary.md"

    # Legacy paths（舊版 models/btc/）
    def legacy_symbol_dir(self, symbol: str) -> Path:
        sym = symbol.strip().lower().replace("usdt", "")
        return self.models_root() / sym

    def metadata_path(self) -> Path:
        return self.models_root() / "model_metadata.json"

    def training_log_path(self) -> Path:
        return self.models_root() / "training_log.json"


def load_combined_features(symbol: str, *, paths: Optional[Phase4Paths] = None) -> pd.DataFrame:
    p = (paths or Phase4Paths.default()).features_combined_path(symbol)
    if not p.is_file():
        raise FileNotFoundError(f"找不到 Phase 2 combined_features：{p}")
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "open_time"
    return df.sort_index()


def load_combined_labels(symbol: str, *, paths: Optional[Phase4Paths] = None) -> pd.DataFrame:
    p = (paths or Phase4Paths.default()).labels_combined_path(symbol)
    if not p.is_file():
        raise FileNotFoundError(f"找不到 Phase 3 combined_labels：{p}")
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "open_time"
    return df.sort_index()


def save_model(
    model,
    symbol: str,
    label_type: str,
    *,
    regime: str = "all_day",
    paths: Optional[Phase4Paths] = None,
) -> Path:
    p4 = paths or Phase4Paths.default()
    p = p4.model_path(symbol, regime, label_type)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    return p


def load_model(
    symbol: str,
    label_type: str,
    *,
    regime: str = "all_day",
    paths: Optional[Phase4Paths] = None,
):
    p4 = paths or Phase4Paths.default()
    p = p4.model_path(symbol, regime, label_type)
    if not p.is_file():
        raise FileNotFoundError(f"找不到模型檔：{p}")
    return joblib.load(p)


def save_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def update_json(path: Path, update: dict) -> None:
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            base = json.load(f)
    else:
        base = {}
    base.update(update)
    save_json_atomic(path, base)


def walk_forward_splits(
    index: pd.DatetimeIndex,
    *,
    n_splits: int = 5,
    val_size: float = 0.2,
    purge_minutes: int = PURGE_MINUTES,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward 多折分割（時間序列）+ purge。\n
    - train 在前、val 在後\n
    - 在 train_end 與 val_start 附近移除 purge_minutes（避免 lookahead leakage）\n

    Returns:
        迭代產生 (train_idx, val_idx)，皆為整數位置索引（numpy array）。
    """
    if len(index) < 10_000:
        # 太短的序列做多折沒有意義（仍允許，但提醒）
        pass
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("walk_forward_splits 需要 DatetimeIndex")

    idx = index.sort_values()
    n = len(idx)
    val_n = max(1, int(n * val_size))
    # 讓每折的 val 區間向後滑動，train 區間逐步擴張
    step = max(1, int((n - val_n) / n_splits))
    purge_td = pd.Timedelta(minutes=purge_minutes)

    for i in range(n_splits):
        train_end_pos = min(n - val_n - 1, (i + 1) * step)
        val_start_pos = train_end_pos + 1
        val_end_pos = min(n - 1, val_start_pos + val_n - 1)

        train_end_time = idx[train_end_pos]
        val_start_time = idx[val_start_pos]

        # purge：train 去掉最後 purge_td；val 去掉最前 purge_td
        train_mask = idx <= (train_end_time - purge_td)
        val_mask = (idx >= (val_start_time + purge_td)) & (idx <= idx[val_end_pos])

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        yield train_idx, val_idx

