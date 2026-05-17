# -*- coding: utf-8 -*-
"""
Phase 2 共用工具（Feature Utils）。

本檔案負責：
- 從 Data_Lake 讀取 Phase 1 原始 1m K 線（統一入口）
- 計算共用的序列（log return、price change）
- 以增量 checkpoint 方式寫入/讀取特徵檔（Parquet）

資料流（你已選擇的規範）：
- 輸入：Data_Lake raw（透過 `utils.data_loader.load_raw_klines`）
- 輸出：Data_Lake processed/features

注意：
- NaN 規則：窗口不足的前 W-1 筆必為 NaN（各指標函數負責）
- 本模組不在 import 時建立任何資料夾；只有在 save/load 時才會 mkdir
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.config import PARQUET_DATE_RANGE_LABEL, get_data_lake_root
from utils.data_loader import load_raw_klines as _load_raw_from_lake

# Phase 2 Multi-timeframe：5 種回顧窗口（分鐘）
WINDOWS: tuple[int, ...] = (50, 100, 240, 480, 720)
NUM_WINDOWS = len(WINDOWS)
FEATURES_PER_SYMBOL = 5 * NUM_WINDOWS  # 5 指標 × 5 窗口 = 25


@dataclass(frozen=True)
class FeaturePaths:
    """集中管理 Phase 2 的輸出路徑。"""

    root: Path  # .../processed/crypto_microstructure/features

    @staticmethod
    def default() -> "FeaturePaths":
        root = get_data_lake_root() / "processed" / "crypto_microstructure" / "features"
        return FeaturePaths(root=root)

    def symbol_dir(self, symbol: str) -> Path:
        return self.root / symbol.strip().upper()

    def indicator_path(self, symbol: str, indicator: str) -> Path:
        name = indicator.strip().lower()
        return self.symbol_dir(symbol) / f"{name}.parquet"

    def combined_path(self, symbol: str) -> Path:
        return self.symbol_dir(symbol) / "combined_features.parquet"

    def metadata_path(self) -> Path:
        return self.root / "feature_metadata.json"

    def run_log_path(self) -> Path:
        return self.root / "feature_engineering_log.json"


def load_raw_klines(symbol: str) -> pd.DataFrame:
    """
    從 Data_Lake 讀取 Phase 1 的原始 K 線（1m, 2021-2026）。

    Returns:
        DataFrame：索引 `open_time`（DatetimeIndex），欄位含 open/high/low/close/volume。
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
    計算 log return：r_t = log(P_t) - log(P_{t-1})

    Notes:
        - 若 close <= 0，log 不存在，會產生 NaN；這代表輸入資料異常（Phase 1 應已清洗）。
    """
    close = df["close"].astype("float64")
    r = np.log(close).diff()
    r.name = "log_return"
    return r


def compute_price_changes(df: pd.DataFrame) -> pd.Series:
    """
    計算價格變化：Δp_t = P_t - P_{t-1}
    """
    dp = df["close"].astype("float64").diff()
    dp.name = "price_change"
    return dp


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_feature(df: pd.DataFrame, symbol: str, indicator: str, *, paths: Optional[FeaturePaths] = None) -> Path:
    """
    儲存單一指標特徵（checkpoint），固定寫成 Parquet（zstd 壓縮）。

    Args:
        df: 欄位為該指標計算結果（通常含 5 窗口欄位，如 roll_50…roll_720），索引需與 raw 對齊。
        symbol: 交易對。
        indicator: 指標名（例如 'roll', 'amihud', 'vpin'）。
        paths: 可選，覆寫輸出根目錄（測試用）。
    """
    p = (paths or FeaturePaths.default()).indicator_path(symbol, indicator)
    _ensure_parent(p)
    df.to_parquet(p, index=True, compression="zstd", compression_level=9)
    return p


def load_feature(symbol: str, indicator: str, *, paths: Optional[FeaturePaths] = None) -> pd.DataFrame:
    """
    讀取單一指標特徵（checkpoint）。

    Raises:
        FileNotFoundError: 找不到檔案時拋出（上層可用於斷點續傳判斷）。
    """
    p = (paths or FeaturePaths.default()).indicator_path(symbol, indicator)
    if not p.is_file():
        raise FileNotFoundError(f"找不到特徵檔案：{p}")
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "open_time"
    return df.sort_index()


def feature_exists(symbol: str, indicator: str, *, paths: Optional[FeaturePaths] = None) -> bool:
    """用於斷點續傳：檔案存在就跳過該指標。"""
    p = (paths or FeaturePaths.default()).indicator_path(symbol, indicator)
    return p.is_file()


def update_json(path: Path, update: dict) -> None:
    """
    以「讀取->更新->原子寫入」方式更新 JSON。
    Phase 2 會用於 feature_metadata.json 與 feature_engineering_log.json。
    """
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

