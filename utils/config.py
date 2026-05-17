# -*- coding: utf-8 -*-
"""
專案共用設定：Data_Lake 根路徑、Binance 原始檔目錄、Phase 1 日期與交易對常數。

路徑優先順序：
1. 環境變數 DATA_LAKE_ROOT（建議寫入 .env，由 python-dotenv 載入）
2. 預設 C:\\Data_Lake（你已決定統一放 C 槽）

注意：不在模組 import 時自動建立資料夾，請呼叫 ensure_raw_crypto_dir()。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# 專案根目錄（copy/）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 全專案預設根目錄（你已決定統一放 C 槽）
_DEFAULT_DATA_LAKE_ROOT = r"C:\Data_Lake"


def load_project_env() -> dict[str, str | None]:
    """僅從專案根目錄載入 .env（避免 backtest/.env 誤放 token）。"""
    env_path = _PROJECT_ROOT / ".env"
    file_vals = dotenv_values(env_path) if env_path.is_file() else {}
    load_dotenv(env_path, override=True)
    for key, val in file_vals.items():
        if val is not None and key not in os.environ:
            os.environ[key] = val
    return file_vals

# 全專案統一之 K 線區間與輸出檔名年份標籤
START_DATE = "2021-01-01"
END_DATE = "2026-02-28"
PARQUET_DATE_RANGE_LABEL = "2021-2026"
SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")


def get_data_lake_root() -> Path:
    """
    解析並回傳 Data_Lake 根目錄（絕對路徑）。

    每次呼叫會重新讀取環境變數（並再次嘗試載入 .env），方便測試或同一行程內更新路徑。
    """
    load_project_env()
    raw = os.environ.get("DATA_LAKE_ROOT", _DEFAULT_DATA_LAKE_ROOT).strip()
    return Path(raw).expanduser().resolve()


def get_raw_crypto_binance_path() -> Path:
    """Binance 原始 K 線 Parquet 存放目錄：Data_Lake/raw/crypto/binance/"""
    return get_data_lake_root() / "raw" / "crypto" / "binance"


# 相容計畫文件中的命名：多數模組可使用下列屬性（於 import 當下解析一次）
DATA_LAKE_PATH = get_data_lake_root()
RAW_CRYPTO_PATH = DATA_LAKE_PATH / "raw" / "crypto" / "binance"


def ensure_raw_crypto_dir() -> Path:
    """
    確保 Binance 原始資料目錄存在，並回傳該路徑。

    Returns:
        Path: raw/crypto/binance 目錄。
    """
    p = get_raw_crypto_binance_path()
    p.mkdir(parents=True, exist_ok=True)
    return p


def final_parquet_path(
    symbol: str,
    frequency: str = "1m",
    *,
    date_range_label: str | None = None,
) -> Path:
    """
    正式輸出檔路徑，例如 BTCUSDT_1m_2021-2026.parquet。

    Args:
        date_range_label: 覆寫檔名中年份區段（Notebook 小範圍測試用）；預設為 PARQUET_DATE_RANGE_LABEL。
    """
    sym = symbol.strip().upper()
    freq = frequency.strip().lower()
    label = date_range_label if date_range_label is not None else PARQUET_DATE_RANGE_LABEL
    return get_raw_crypto_binance_path() / f"{sym}_{freq}_{label}.parquet"


def download_log_path() -> Path:
    """download_log.json 完整路徑。"""
    return get_raw_crypto_binance_path() / "download_log.json"
