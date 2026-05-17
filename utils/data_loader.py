# -*- coding: utf-8 -*-
"""
Data_Lake 原始 K 線讀取模組。

所有從中央資料湖讀取 Binance Parquet 的程式都應透過本模組，
避免在專案各處硬編碼路徑，方便之後換磁碟或換機器。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.config import get_data_lake_root


def _raw_binance_dir(root: Optional[Path] = None) -> Path:
    """原始 Binance K 線所在目錄：Data_Lake/raw/crypto/binance/"""
    base = root if root is not None else get_data_lake_root()
    return base / "raw" / "crypto" / "binance"


def _filename_pattern(symbol: str, frequency: str, date_range: Optional[str]) -> str:
    """組出預期的檔名（不含路徑）。"""
    sym = symbol.strip().upper()
    freq = frequency.strip().lower()
    if date_range is None:
        return f"{sym}_{freq}_*.parquet"
    dr = date_range.strip()
    return f"{sym}_{freq}_{dr}.parquet"


def _list_matching_parquet(
    directory: Path,
    symbol: str,
    frequency: str,
    date_range: Optional[str],
) -> list[Path]:
    """
    在 directory 內列出符合命名慣例的 Parquet 檔案。

    命名慣例：{SYMBOL}_{frequency}_{date_range}.parquet
    例如：BTCUSDT_1m_2021-2026.parquet
    """
    sym = symbol.strip().upper()
    freq = frequency.strip().lower()
    if not directory.is_dir():
        return []

    if date_range is not None:
        dr = date_range.strip()
        exact = directory / f"{sym}_{freq}_{dr}.parquet"
        return [exact] if exact.is_file() else []

    # date_range 為 None：找出所有 SYMBOL_freq_*.parquet
    prefix = f"{sym}_{freq}_"
    suffix = ".parquet"
    out: list[Path] = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not name.endswith(suffix):
            continue
        if not name.startswith(prefix):
            continue
        middle = name[len(prefix) : -len(suffix)]
        if "smoke" in middle.lower():
            continue
        # 中間段應像 2021-2026 或 2024-2026（允許英數與連字號）
        if middle and re.fullmatch(r"[\w\-]+", middle):
            out.append(p)
    return sorted(out, key=lambda x: x.name)


def load_raw_klines(
    symbol: str,
    frequency: str = "1m",
    date_range: Optional[str] = None,
    *,
    data_lake_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    從 Data_Lake 讀取指定幣種的原始 K 線 Parquet。

    Args:
        symbol: 交易對，例如 "BTCUSDT"、"ETHUSDT"（大小寫不敏感，內部會轉大寫）。
        frequency: K 線週期，預設 "1m"（與檔名中片段一致）。
        date_range: 檔名中的年份區間字串，例如 "2021-2026"、"2024-2026"。
            若為 None，則在資料夾內尋找唯一一個符合 {SYMBOL}_{frequency}_*.parquet 的檔案；
            若找到 0 個或多個，會拋出清楚的中文錯誤說明。
        data_lake_root: 可選，強制指定 Data_Lake 根路徑（測試或特殊情境用）。

    Returns:
        pd.DataFrame: Parquet 內容（欄位依 Phase 1 下載程式寫入為準）。

    Raises:
        FileNotFoundError: 目錄不存在、檔案不存在，或無法唯一決定要讀哪一個檔案時，
            以 FileNotFoundError 承載中文訊息，方便上層統一處理。
    """
    root = (data_lake_root or get_data_lake_root()).resolve()
    binance_dir = _raw_binance_dir(root)

    if not binance_dir.is_dir():
        raise FileNotFoundError(
            f"找不到 Binance 原始資料目錄：{binance_dir}\n"
            f"請確認已建立 Data_Lake 結構，或設定正確的 DATA_LAKE_ROOT（目前根目錄：{root}）。"
        )

    matches = _list_matching_parquet(binance_dir, symbol, frequency, date_range)

    if len(matches) == 0:
        hint = _filename_pattern(symbol, frequency, date_range)
        raise FileNotFoundError(
            f"在「{binance_dir}」找不到符合的 Parquet 檔案。\n"
            f"預期可符合的檔名模式：{hint}\n"
            f"交易對：{symbol.strip().upper()}，週期：{frequency.strip().lower()}。\n"
            f"若有多個年份版本的檔案，請在參數 date_range 指定，例如 date_range='2021-2026'。"
        )

    if len(matches) > 1:
        names = "\n".join(f"  - {p.name}" for p in matches)
        raise FileNotFoundError(
            f"找到多個符合的檔案，請用參數 date_range 指定要讀哪一個：\n{names}"
        )

    path = matches[0]
    if not path.is_file():
        raise FileNotFoundError(f"檔案不存在或不是一般檔案：{path}")

    # engine 預設會用 pyarrow；若未安裝可改 fastparquet（本專案 requirements 已列 pyarrow）
    df = pd.read_parquet(path)
    return df


# ---------------------------------------------------------------------------
# 使用範例（僅在直接執行此檔案時示範；實際專案請從其他模組 import）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 範例 1：指定完整檔名中的年份區段
    # df = load_raw_klines("BTCUSDT", frequency="1m", date_range="2021-2026")
    # print(df.head())

    # 範例 2：資料夾內只有一個 BTCUSDT_1m_*.parquet 時可省略 date_range
    # df = load_raw_klines("BTCUSDT", "1m")

    # 範例 3：改用其他磁碟的 Data_Lake（建議用環境變數，不必改程式）
    # import os
    # os.environ["DATA_LAKE_ROOT"] = r"C:\Data_Lake"
    # df = load_raw_klines("ETHUSDT", date_range="2024-2026")

    print("data_loader 模組已載入。")
    print(f"目前解析的 Data_Lake 根目錄：{get_data_lake_root()}")
    print("請將範例程式碼取消註解並準備好 Parquet 檔後再執行讀取。")
