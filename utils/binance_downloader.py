# -*- coding: utf-8 -*-
"""
Binance 公開 API 1 分鐘 K 線下載器。

- 按月分段請求（每請求最多 1000 根），並將完成月份寫入 .checkpoints 以利斷點續傳。
- volume 欄位使用 USDT 計價成交量（API 第 8 欄 quote asset volume，索引 7）。
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from tqdm import tqdm

from utils.config import (
    PARQUET_DATE_RANGE_LABEL,
    download_log_path,
    ensure_raw_crypto_dir,
    final_parquet_path,
)

# Binance kline 陣列索引（見官方文件）
_K_OPEN_TIME = 0
_K_OPEN = 1
_K_HIGH = 2
_K_LOW = 3
_K_CLOSE = 4
_K_QUOTE_ASSET_VOLUME = 7

# 公開端點請求間隔（秒），降低觸發限流機率
_REQUEST_SLEEP_SEC = 0.2
_MAX_BATCH = 1000


def _utc_ts_start_end(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    將日期字串轉為 UTC 時間範圍（含首尾分鐘 K 線）。

    start_date: 當日 00:00:00 UTC
    end_date: 當日最後一分鐘 23:59:00 UTC（1m K 線 open_time）
    """
    start = pd.Timestamp(f"{start_date} 00:00:00", tz="UTC")
    end_day = pd.Timestamp(f"{end_date} 23:59:00", tz="UTC")
    return start, end_day


def _month_periods_covering(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    產生 [start, end] 內每個曆月的 (month_start, month_end_open_time, key)。

    month_end_open_time 為該月最後一根 1m K 的 open_time（對齊 23:59）。
    """
    out: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    last = end
    while cur <= last:
        if cur.month == 12:
            nxt = pd.Timestamp(year=cur.year + 1, month=1, day=1, tz="UTC")
        else:
            nxt = pd.Timestamp(year=cur.year, month=cur.month + 1, day=1, tz="UTC")
        month_end_open = nxt - pd.Timedelta(minutes=1)
        key = f"{cur.year:04d}-{cur.month:02d}"
        out.append((cur, month_end_open, key))
        cur = nxt
    return out


def _klines_to_dataframe(rows: list[list[Any]]) -> pd.DataFrame:
    """將 Binance 原始 kline 陣列轉成標準 DataFrame（索引為 open_time）。"""
    records: list[dict[str, Any]] = []
    for k in rows:
        ot = pd.to_datetime(k[_K_OPEN_TIME], unit="ms", utc=True)
        records.append(
            {
                "open_time": ot,
                "open": float(k[_K_OPEN]),
                "high": float(k[_K_HIGH]),
                "low": float(k[_K_LOW]),
                "close": float(k[_K_CLOSE]),
                "volume": float(k[_K_QUOTE_ASSET_VOLUME]),
            }
        )
    df = pd.DataFrame.from_records(records)
    df = df.set_index("open_time").sort_index()
    df.index.name = "open_time"
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    return df


def _load_download_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"無法解析 download_log.json（JSON 格式錯誤）：{path}\n"
            f"詳情：{e}\n建議：請備份後刪除此檔，或手動修復內容。"
        ) from e


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _session_storage_key(symbol: str, session_tag: str | None) -> str:
    """正式下載用交易對；測試／Notebook 可加 session_tag 隔離 checkpoint 與 log。"""
    sym = symbol.strip().upper()
    if session_tag is None or not str(session_tag).strip():
        return sym
    return f"{sym}__{str(session_tag).strip()}"


def _checkpoint_dir(symbol: str, session_tag: str | None) -> Path:
    """斷點目錄：.checkpoints/{SYMBOL} 或 .checkpoints/{SYMBOL}__{tag}。"""
    key = _session_storage_key(symbol, session_tag)
    d = ensure_raw_crypto_dir() / ".checkpoints" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clear_symbol_progress(symbol: str, log: dict[str, Any], session_tag: str | None) -> None:
    """清除某交易對（或測試工作階段）的 checkpoint 與 log 條目（不先建立 checkpoint 目錄）。"""
    key = _session_storage_key(symbol, session_tag)
    cp = ensure_raw_crypto_dir() / ".checkpoints" / key
    if cp.is_dir():
        shutil.rmtree(cp, ignore_errors=True)
    log.pop(key, None)


def _download_month_klines(
    client: Client,
    symbol: str,
    month_start: pd.Timestamp,
    month_end_open: pd.Timestamp,
    overall_start: pd.Timestamp,
    overall_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    下載單一曆月範圍內的 1m K 線（裁剪到全專案起訖時間內）。
    """
    sym = symbol.upper()
    req_start = max(month_start, overall_start)
    req_end_open = min(month_end_open, overall_end)
    if req_start > req_end_open:
        return pd.DataFrame()

    start_ms = int(req_start.timestamp() * 1000)
    end_ms = int(req_end_open.timestamp() * 1000)
    all_rows: list[list[Any]] = []
    cursor = start_ms

    while cursor <= end_ms:
        time.sleep(_REQUEST_SLEEP_SEC)
        try:
            batch = client.get_klines(
                symbol=sym,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                startTime=cursor,
                endTime=end_ms,
                limit=_MAX_BATCH,
            )
        except (BinanceAPIException, BinanceRequestException) as e:
            raise RuntimeError(
                f"從 Binance 取得 K 線失敗（{sym}）。\n"
                f"錯誤：{e}\n"
                f"建議：檢查網路連線、交易對代號是否正確，或稍後再試（可能觸發限流）。"
            ) from e

        if not batch:
            break
        all_rows.extend(batch)
        last_open = batch[-1][0]
        cursor = int(last_open) + 60_000
        if len(batch) < _MAX_BATCH:
            break

    if not all_rows:
        return pd.DataFrame()

    df = _klines_to_dataframe(all_rows)
    df = df[(df.index >= overall_start) & (df.index <= overall_end)]
    return df


def download_binance_klines(
    symbol: str,
    start_date: str = "2021-01-01",
    end_date: str = "2026-02-28",
    force_redownload: bool = False,
    *,
    session_tag: str | None = None,
) -> pd.DataFrame:
    """
    下載 Binance 1 分鐘 K 線，合併為完整 DataFrame，並寫入正式 Parquet（未清洗版）。

    已完成的月份會快取於 raw/crypto/binance/.checkpoints/{SYMBOL}/YYYY-MM.parquet；
    download_log.json 記錄各月狀態。

    Args:
        symbol: 交易對，例如 BTCUSDT。
        start_date: 起始日期（UTC 日期字串）。
        end_date: 結束日期（UTC 日期字串，含當日）。
        force_redownload: True 時清除該幣種（或測試工作階段）checkpoint 與 log 後重新下載。
        session_tag: 非 None 時，checkpoint 與 log 使用獨立命名空間，且 Parquet 檔名中年份標籤改為此字串
            （Notebook 小範圍測試建議傳例如 \"notebook\"，避免覆寫正式檔與斷點狀態）。

    Returns:
        合併、排序、去重並裁剪時間範圍後之 DataFrame（索引 open_time）。

    Raises:
        RuntimeError: API 或檔案系統相關錯誤（中文訊息）。
    """
    ensure_raw_crypto_dir()
    sym = symbol.strip().upper()
    storage_key = _session_storage_key(sym, session_tag)
    parquet_label = PARQUET_DATE_RANGE_LABEL if session_tag is None else str(session_tag).strip()
    overall_start, overall_end = _utc_ts_start_end(start_date, end_date)
    log_path = download_log_path()
    log = _load_download_log(log_path)

    if force_redownload:
        _clear_symbol_progress(sym, log, session_tag)
        _atomic_write_json(log_path, log)

    if storage_key not in log:
        log[storage_key] = {"symbol": sym, "session_tag": session_tag, "months": {}, "last_updated": None}

    client = Client("", "")
    cp_root = _checkpoint_dir(sym, session_tag)
    months = _month_periods_covering(overall_start, overall_end)

    for month_start, month_end_open, key in tqdm(
        months,
        desc=f"下載 {storage_key} 按月",
        unit="月",
    ):
        month_state = log[storage_key].setdefault("months", {})
        ck_file = cp_root / f"{key}.parquet"

        if not force_redownload and key in month_state and month_state[key] == "completed" and ck_file.is_file():
            continue

        df_m = _download_month_klines(
            client, sym, month_start, month_end_open, overall_start, overall_end
        )
        if df_m.empty:
            month_state[key] = "empty"
            if ck_file.is_file():
                ck_file.unlink()
        else:
            df_m.to_parquet(ck_file, index=True)
            month_state[key] = "completed"

        log[storage_key]["last_updated"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(log_path, log)

    # 合併所有已完成月份之 checkpoint
    dfs: list[pd.DataFrame] = []
    for _, _, key in months:
        ck = cp_root / f"{key}.parquet"
        if ck.is_file():
            part = pd.read_parquet(ck)
            if not isinstance(part.index, pd.DatetimeIndex):
                part.index = pd.to_datetime(part.index, utc=True)
            if part.index.tz is None:
                part.index = part.index.tz_localize("UTC")
            part.index.name = "open_time"
            dfs.append(part)

    if not dfs:
        raise RuntimeError(
            f"合併失敗：找不到任何月份 checkpoint 檔案（{storage_key}）。\n"
            f"目錄：{cp_root}\n建議：設定 force_redownload=True 重新下載，或檢查網路與日期區間。"
        )

    merged = pd.concat(dfs).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged[(merged.index >= overall_start) & (merged.index <= overall_end)]

    out_path = final_parquet_path(sym, "1m", date_range_label=parquet_label)
    try:
        merged.to_parquet(out_path, index=True)
    except OSError as e:
        raise RuntimeError(
            f"無法寫入 Parquet：{out_path}\n錯誤：{e}\n"
            f"建議：確認 DATA_LAKE_ROOT 指向有效磁碟且有足夠空間與寫入權限。"
        ) from e

    return merged
