# -*- coding: utf-8 -*-
"""
Phase 2 增量主控特徵工程管線（Multi-timeframe）。

目標：
- 依序處理 BTCUSDT / ETHUSDT
- 依序計算 5 個指標 × 5 窗口（50/100/240/480/720）
- 每個指標計算完立即存檔（checkpoint），可斷點續傳
- 全部完成後合併為 combined_features.parquet（25 欄）
- 產生 feature_metadata.json 與 feature_engineering_log.json

資料規範：
- 輸入：Data_Lake raw（{DATA_LAKE_ROOT}/raw/crypto/binance/{symbol}_1m_2021-2026.parquet）
- 輸出：Data_Lake processed/features
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from features.amihud_measure import compute_amihud_features
from features.feature_utils import (
    FEATURES_PER_SYMBOL,
    NUM_WINDOWS,
    WINDOWS,
    FeaturePaths,
    feature_exists,
    load_feature,
    load_raw_klines,
    save_feature,
    update_json,
)
from features.kyle_lambda import compute_kyle_lambda_features
from features.roll_measure import compute_roll_features, compute_roll_impact_features
from features.vpin import compute_vpin_features
from utils.config import SYMBOLS

INDICATOR_COLUMN_PREFIX: dict[str, str] = {
    "roll": "roll_",
    "roll_impact": "roll_impact_",
    "amihud": "amihud_",
    "kyle_lambda": "kyle_lambda_",
    "vpin": "vpin_",
}


def _expected_columns(indicator: str) -> list[str]:
    prefix = INDICATOR_COLUMN_PREFIX[indicator]
    return [f"{prefix}{w}" for w in WINDOWS]


def _nan_ratio(df: pd.DataFrame) -> dict[str, float]:
    return {c: float(df[c].isna().mean()) for c in df.columns}


def _basic_stats(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "cols": list(df.columns),
        "nan_ratio": _nan_ratio(df),
        "mean": {c: float(df[c].mean(skipna=True)) for c in df.columns},
        "std": {c: float(df[c].std(skipna=True)) for c in df.columns},
        "min": {c: float(df[c].min(skipna=True)) for c in df.columns},
        "max": {c: float(df[c].max(skipna=True)) for c in df.columns},
    }


def _feature_is_complete(df: pd.DataFrame, indicator: str) -> bool:
    expected = _expected_columns(indicator)
    if len(df.columns) != len(expected):
        return False
    return list(df.columns) == expected


def _compute_or_load(
    *,
    symbol: str,
    indicator: str,
    compute_fn: Callable[[pd.DataFrame], pd.DataFrame],
    raw_df: pd.DataFrame,
    paths: FeaturePaths,
    force_recompute: bool,
) -> tuple[pd.DataFrame, str]:
    """
    回傳 (df, status)：
    - status = 'computed' | 'skipped' | 'recomputed_stale'
    """
    expected = _expected_columns(indicator)
    stale_checkpoint = False

    if (not force_recompute) and feature_exists(symbol, indicator, paths=paths):
        df = load_feature(symbol, indicator, paths=paths)
        if _feature_is_complete(df, indicator):
            return df, "skipped"
        stale_checkpoint = True
        print(
            f"⚠️ {symbol} - {indicator} checkpoint 欄位不符（預期 {len(expected)} 欄），將重新計算..."
        )

    print(f"正在計算 {symbol} {indicator}（窗口 {list(WINDOWS)}）...")
    df = compute_fn(raw_df)
    if not _feature_is_complete(df, indicator):
        missing = [c for c in expected if c not in df.columns]
        raise ValueError(f"{symbol} {indicator} 計算結果不完整，缺少欄位：{missing}")

    save_feature(df, symbol, indicator, paths=paths)
    status = "recomputed_stale" if stale_checkpoint else "computed"
    return df, status


def run_feature_pipeline(force_recompute: bool = False) -> None:
    """
    依序計算並存檔所有指標，最後合併為 combined_features.parquet（每幣種 25 欄）。
    """
    paths = FeaturePaths.default()

    run_started = datetime.now(timezone.utc).isoformat()
    run_log: dict = {
        "run_started_utc": run_started,
        "force_recompute": bool(force_recompute),
        "symbols": list(SYMBOLS),
        "windows": list(WINDOWS),
        "features_per_symbol": FEATURES_PER_SYMBOL,
        "steps": [],
    }

    for symbol in SYMBOLS:
        print(f"\n========== {symbol} ==========")
        raw_df = load_raw_klines(symbol)

        step_defs: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
            ("roll", compute_roll_features),
            ("roll_impact", compute_roll_impact_features),
            ("amihud", compute_amihud_features),
            ("kyle_lambda", compute_kyle_lambda_features),
            ("vpin", compute_vpin_features),
        ]

        computed_frames: list[pd.DataFrame] = []
        for indicator, fn in step_defs:
            t0 = time.time()
            df_i, status = _compute_or_load(
                symbol=symbol,
                indicator=indicator,
                compute_fn=fn,
                raw_df=raw_df,
                paths=paths,
                force_recompute=force_recompute,
            )
            elapsed = time.time() - t0

            computed_frames.append(df_i)
            status_msg = {
                "computed": "計算並儲存",
                "recomputed_stale": "重新計算並儲存（舊 checkpoint 已過期）",
                "skipped": "存在並載入/跳過",
            }[status]
            print(f"✅ {symbol} - {indicator}（{NUM_WINDOWS} 窗口）已{status_msg}")

            run_log["steps"].append(
                {
                    "symbol": symbol,
                    "indicator": indicator,
                    "status": status,
                    "elapsed_sec": float(elapsed),
                    "columns": list(df_i.columns),
                    "nan_ratio": _nan_ratio(df_i),
                }
            )

            update_json(paths.run_log_path(), {run_started: run_log})
            update_json(
                paths.metadata_path(),
                {
                    f"{symbol}.{indicator}": {
                        "windows": list(WINDOWS),
                        "saved_path": str(paths.indicator_path(symbol, indicator)),
                        "stats": _basic_stats(df_i),
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                    }
                },
            )

        combined = pd.concat(computed_frames, axis=1)
        combined = combined.sort_index()

        if len(combined.columns) != FEATURES_PER_SYMBOL:
            raise ValueError(
                f"{symbol} combined_features 欄數錯誤："
                f"預期 {FEATURES_PER_SYMBOL}，實際 {len(combined.columns)}"
            )

        combined_path = paths.combined_path(symbol)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(combined_path, index=True, compression="zstd", compression_level=9)

        print(f"✅ {symbol} - combined_features（{len(combined.columns)} 欄）已儲存：{combined_path}")

        update_json(
            paths.metadata_path(),
            {
                f"{symbol}.combined": {
                    "windows": list(WINDOWS),
                    "feature_count": len(combined.columns),
                    "saved_path": str(combined_path),
                    "stats": _basic_stats(combined),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                }
            },
        )

    run_log["run_ended_utc"] = datetime.now(timezone.utc).isoformat()
    update_json(paths.run_log_path(), {run_started: run_log})
    print("\nPhase 2 特徵管線執行完畢。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2：Feature Engineering 增量管線（Multi-timeframe）")
    parser.add_argument("--force", action="store_true", help="強制重算並覆寫所有特徵 checkpoint")
    args = parser.parse_args()
    run_feature_pipeline(force_recompute=args.force)
