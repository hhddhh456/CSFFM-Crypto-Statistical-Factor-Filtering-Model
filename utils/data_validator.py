# -*- coding: utf-8 -*-
"""
K 線資料清洗與品質驗證。

與 Phase 1 規格對齊：1 分鐘頻率、全專案日期範圍重索引、OHLC 前向填補、volume 缺值填 0。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.config import END_DATE, PARQUET_DATE_RANGE_LABEL, START_DATE
from utils.data_loader import load_raw_klines

def _project_utc_bounds(
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    起訖時間（含首尾分鐘 K 線 open_time）。

    未指定時使用專案預設 START_DATE / END_DATE。
    """
    sd = start_date or START_DATE
    ed = end_date or END_DATE
    start = pd.Timestamp(f"{sd} 00:00:00", tz="UTC")
    end = pd.Timestamp(f"{ed} 23:59:00", tz="UTC")
    return start, end


def clean_klines(
    df: pd.DataFrame,
    symbol: str,
    *,
    reindex_start_date: str | None = None,
    reindex_end_date: str | None = None,
) -> pd.DataFrame:
    """
    清洗 K 線：排序、移除價格異常、對指定（或全專案）區間做 1 分鐘重索引並填補缺口。

    Args:
        df: 索引須為 open_time（DatetimeIndex，建議 UTC）；欄位 open, high, low, close, volume。
        symbol: 交易對代號（僅用於未來擴充記錄／除錯，目前不影響邏輯）。
        reindex_start_date: 1 分鐘重索引起始日期（UTC 日期字串）；預設為專案 START_DATE。
        reindex_end_date: 重索引結束日期；預設為專案 END_DATE。Notebook 小範圍測試請與下載區間一致。

    Returns:
        清洗後 DataFrame（索引名稱 open_time，欄位 dtype float64）。
    """
    _ = symbol  # 保留參數以利與 SOP 介面一致
    if df.empty:
        raise ValueError("clean_klines：輸入 DataFrame 為空，無法清洗。")

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "open_time"
    out = out.sort_index()

    for c in ("open", "high", "low", "close", "volume"):
        if c not in out.columns:
            raise ValueError(f"clean_klines：缺少必要欄位「{c}」。")

    bad = (
        (out["open"] <= 0)
        | (out["high"] <= 0)
        | (out["low"] <= 0)
        | (out["close"] <= 0)
        | (out["volume"] < 0)
    )
    out = out[~bad]
    if out.empty:
        raise ValueError("clean_klines：移除異常價格後資料為空，請檢查下載來源是否正確。")

    full_start, full_end = _project_utc_bounds(reindex_start_date, reindex_end_date)
    full_idx = pd.date_range(full_start, full_end, freq="1min", tz="UTC")
    out = out.reindex(full_idx)
    out.index.name = "open_time"

    ohlc = ["open", "high", "low", "close"]
    out[ohlc] = out[ohlc].ffill()
    out[ohlc] = out[ohlc].bfill()
    out["volume"] = out["volume"].fillna(0.0)

    for c in ("open", "high", "low", "close", "volume"):
        out[c] = out[c].astype("float64")

    return out


def validate_klines(
    symbol: str,
    *,
    date_range: str | None = None,
    use_project_expected_length: bool = True,
) -> dict[str, Any]:
    """
    讀取 Data_Lake 中已存檔之 Parquet 並產出品質報告。

    Args:
        symbol: 交易對，例如 BTCUSDT。
        date_range: 檔名年份區段；預設為專案統一之 PARQUET_DATE_RANGE_LABEL。
        use_project_expected_length: True 時以全專案日期區間計算預期分鐘數；Notebook 小檔可改 False，改以檔案內最小至最大時間計算預期長度。

    Returns:
        字典（中文鍵名），含筆數、時間範圍、缺失率、重複索引數等。
    """
    label = date_range if date_range is not None else PARQUET_DATE_RANGE_LABEL
    try:
        df = load_raw_klines(symbol, "1m", label)
    except FileNotFoundError as e:
        return {
            "成功": False,
            "訊息": str(e),
            "建議": "請先執行 data_pipeline.py 完成下載，或確認 date_range 與檔名一致。",
        }

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    actual_len = int(len(df))
    if use_project_expected_length:
        full_start, full_end = _project_utc_bounds()
        expected_len = len(pd.date_range(full_start, full_end, freq="1min", tz="UTC"))
    elif actual_len > 0:
        expected_len = len(
            pd.date_range(df.index.min(), df.index.max(), freq="1min", tz="UTC")
        )
    else:
        expected_len = 0

    dup = int(df.index.duplicated().sum())
    missing_close = int(df["close"].isna().sum()) if "close" in df.columns else int(len(df))
    abnormal_price = int(((df["close"] <= 0) | (df["open"] <= 0)).sum()) if "open" in df.columns else 0

    missing_rate = missing_close / actual_len if actual_len else 1.0

    return {
        "成功": True,
        "交易對": symbol.strip().upper(),
        "檔名年份標籤": label,
        "總筆數": actual_len,
        "預期筆數_全專案區間": expected_len,
        "時間起_UTC": str(df.index.min()) if actual_len else None,
        "時間訖_UTC": str(df.index.max()) if actual_len else None,
        "重複時間戳筆數": dup,
        "收盤價缺失筆數": missing_close,
        "價格小於等於零筆數": abnormal_price,
        "收盤價缺失率": float(missing_rate),
        "訊息": "驗證完成。若預期筆數與總筆數差異大，請檢查下載是否中斷或未清洗重索引。",
    }
