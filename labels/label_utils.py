# -*- coding: utf-8 -*-
"""
Phase 3 共用工具（Label Utils）。

你目前的資料規範：
- 輸入：Data_Lake raw（透過 `utils.data_loader.load_raw_klines`）
- 輸出：Data_Lake processed labels（C:\\Data_Lake\\processed\\crypto_microstructure\\labels\\...）

核心參數（不可更改）：
- h = 1500
- state_window = 1500

與 Phase 2 對齊：
- `combined_labels` 時間索引必須與 Phase 2 `combined_features.parquet` 完全一致
- 透過 `load_feature_index` / `align_series_to_features` / `assert_index_matches_features`

增量 checkpoint：
- 每種 label 計算完立刻存檔，存在就跳過（除非 force_recompute=True）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from features.feature_utils import FeaturePaths
from utils.config import PARQUET_DATE_RANGE_LABEL, get_data_lake_root
from utils.data_loader import load_raw_klines as _load_raw_from_lake

# Phase 3 固定參數
HORIZON_H = 1500
STATE_WINDOW = 1500


@dataclass(frozen=True)
class LabelPaths:
    root: Path  # .../processed/crypto_microstructure/labels

    @staticmethod
    def default() -> "LabelPaths":
        root = get_data_lake_root() / "processed" / "crypto_microstructure" / "labels"
        return LabelPaths(root=root)

    def symbol_dir(self, symbol: str) -> Path:
        return self.root / symbol.strip().upper()

    def label_path(self, symbol: str, label_type: str) -> Path:
        name = label_type.strip().lower()
        return self.symbol_dir(symbol) / f"{name}.parquet"

    def combined_path(self, symbol: str) -> Path:
        return self.symbol_dir(symbol) / "combined_labels.parquet"

    def metadata_path(self) -> Path:
        return self.root / "label_metadata.json"

    def run_log_path(self) -> Path:
        return self.root / "label_engineering_log.json"


def load_feature_index(symbol: str, *, feature_paths: Optional[FeaturePaths] = None) -> pd.DatetimeIndex:
    """
    讀取 Phase 2 combined_features 的時間索引（標籤對齊基準）。

    Raises:
        FileNotFoundError: Phase 2 尚未產出 combined_features 時拋出。
    """
    p = (feature_paths or FeaturePaths.default()).combined_path(symbol)
    if not p.is_file():
        raise FileNotFoundError(
            f"找不到 Phase 2 combined_features：{p}\n"
            "請先執行：python feature_pipeline.py --force"
        )
    df = pd.read_parquet(p, columns=[])
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx.name = "open_time"
    return idx.sort_values()


def align_series_to_features(
    series: pd.Series,
    symbol: str,
    *,
    feature_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """將 label Series reindex 至 Phase 2 特徵索引（同序、同長）。"""
    idx = feature_index if feature_index is not None else load_feature_index(symbol)
    out = series.reindex(idx)
    out.name = series.name
    return out


def assert_index_matches_features(
    index: pd.Index,
    symbol: str,
    *,
    feature_index: pd.DatetimeIndex | None = None,
) -> None:
    """合併前後驗證索引；不符時 raise ValueError。"""
    feat_idx = feature_index if feature_index is not None else load_feature_index(symbol)
    if not index.equals(feat_idx):
        raise ValueError(
            f"{symbol} 標籤索引與 Phase 2 combined_features 不一致："
            f"labels={len(index)}, features={len(feat_idx)}"
        )


def load_raw_klines_for_labels(symbol: str) -> pd.DataFrame:
    """
    從 Data_Lake 讀取 Phase 1 原始 K 線（1m, 2021-2026），供 Phase 3 使用。
    """
    df = _load_raw_from_lake(symbol, "1m", PARQUET_DATE_RANGE_LABEL)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "open_time"
    return df.sort_index()


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """
    log return：r_t = log(P_t) - log(P_{t-1})
    """
    close = df["close"].astype("float64")
    r = np.log(close).diff()
    r.name = "log_return"
    return r


def shift_forward(series: pd.Series, h: int = HORIZON_H) -> pd.Series:
    """
    將序列往「未來」平移：future(t) = series(t + h)
    """
    return series.shift(-h)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_label(df: pd.DataFrame, symbol: str, label_type: str, *, paths: Optional[LabelPaths] = None) -> Path:
    p = (paths or LabelPaths.default()).label_path(symbol, label_type)
    _ensure_parent(p)
    df.to_parquet(p, index=True, compression="zstd", compression_level=9)
    return p


def load_label(symbol: str, label_type: str, *, paths: Optional[LabelPaths] = None) -> pd.DataFrame:
    p = (paths or LabelPaths.default()).label_path(symbol, label_type)
    if not p.is_file():
        raise FileNotFoundError(f"找不到標籤檔案：{p}")
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "open_time"
    return df.sort_index()


def label_exists(symbol: str, label_type: str, *, paths: Optional[LabelPaths] = None) -> bool:
    p = (paths or LabelPaths.default()).label_path(symbol, label_type)
    return p.is_file()


def label_from_states(current_state: pd.Series, future_state: pd.Series) -> pd.Series:
    """
    label = sign(future - current)，若相等視為 -1。
    """
    diff = future_state - current_state
    out = pd.Series(np.where(diff > 0, 1, -1), index=current_state.index)
    out = out.astype("int8")
    out[current_state.isna() | future_state.isna()] = pd.NA
    return out


def apply_initial_nan_mask(label: pd.Series, *, state_window: int = STATE_WINDOW, h: int = HORIZON_H) -> pd.Series:
    """
    依 SOP 保守處理：前 (state_window + h) 筆設為 NaN。
    """
    out = label.copy()
    n = len(out)
    k = min(state_window + h, n)
    out.iloc[:k] = pd.NA
    return out


def update_json(path: Path, update: dict) -> None:
    _ensure_parent(path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            base = json.load(f)
    else:
        base = {}
    base.update(update)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

