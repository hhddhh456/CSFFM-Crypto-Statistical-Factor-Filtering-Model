# -*- coding: utf-8 -*-
"""
Phase 3 增量主控標籤工程管線（配合 Multi-timeframe）。

特性：
- 依序處理 BTCUSDT / ETHUSDT
- 標籤時間索引與 Phase 2 combined_features 嚴格對齊
- 每完成一種標籤就立即 checkpoint 存檔
- 產生 label_metadata.json 與 label_engineering_log.json
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from labels.jarque_bera import compute_jarque_bera_label
from labels.kurtosis import compute_kurtosis_label
from labels.label_utils import (
    HORIZON_H,
    STATE_WINDOW,
    LabelPaths,
    align_series_to_features,
    assert_index_matches_features,
    label_exists,
    load_feature_index,
    load_label,
    load_raw_klines_for_labels,
    save_label,
    update_json,
)
from labels.realized_volatility import compute_realized_volatility_label
from labels.sequential_correlation import compute_sequential_correlation_label
from labels.skewness import compute_skewness_label
from utils.config import SYMBOLS

FEATURE_ALIGNMENT = "phase2_combined_features"


def _pos_neg_ratio(s: pd.Series) -> dict[str, float]:
    valid = s.dropna()
    if len(valid) == 0:
        return {"pos_ratio": 0.0, "neg_ratio": 0.0}
    pos = float((valid == 1).mean())
    neg = float((valid == -1).mean())
    return {"pos_ratio": pos, "neg_ratio": neg}


def _nan_ratio(s: pd.Series) -> float:
    return float(s.isna().mean())


def _compute_or_load(
    *,
    symbol: str,
    label_type: str,
    compute_fn: Callable[[pd.DataFrame], pd.Series],
    raw_df: pd.DataFrame,
    paths: LabelPaths,
    force_recompute: bool,
    feature_index: pd.DatetimeIndex,
) -> tuple[pd.Series, str]:
    if (not force_recompute) and label_exists(symbol, label_type, paths=paths):
        df = load_label(symbol, label_type, paths=paths)
        col = df.columns[0]
        s = align_series_to_features(df[col], symbol, feature_index=feature_index)
        out = pd.DataFrame({s.name: s})
        save_label(out, symbol, label_type, paths=paths)
        return s, "skipped_realigned"

    s = compute_fn(raw_df)
    s = align_series_to_features(s, symbol, feature_index=feature_index)
    out = pd.DataFrame({s.name: s})
    save_label(out, symbol, label_type, paths=paths)
    return s, "computed"


def run_label_pipeline(force_recompute: bool = False) -> None:
    paths = LabelPaths.default()

    run_started = datetime.now(timezone.utc).isoformat()
    run_log: dict = {
        "run_started_utc": run_started,
        "force_recompute": bool(force_recompute),
        "h": HORIZON_H,
        "state_window": STATE_WINDOW,
        "feature_alignment": FEATURE_ALIGNMENT,
        "symbols": list(SYMBOLS),
        "steps": [],
    }

    for symbol in SYMBOLS:
        print(f"\n========== {symbol}（對齊 Phase 2 索引）==========")
        feature_index = load_feature_index(symbol)
        raw_df = load_raw_klines_for_labels(symbol)

        step_defs: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
            ("realized_volatility", lambda d: compute_realized_volatility_label(d)),
            ("sequential_correlation", lambda d: compute_sequential_correlation_label(d)),
            ("skewness", lambda d: compute_skewness_label(d)),
            ("kurtosis", lambda d: compute_kurtosis_label(d)),
            ("jarque_bera", lambda d: compute_jarque_bera_label(d)),
        ]

        label_series: list[pd.Series] = []
        for label_type, fn in step_defs:
            print(f"正在計算 {symbol} {label_type}（對齊 Phase 2 索引）...")
            t0 = time.time()
            s, status = _compute_or_load(
                symbol=symbol,
                label_type=label_type,
                compute_fn=fn,
                raw_df=raw_df,
                paths=paths,
                force_recompute=force_recompute,
                feature_index=feature_index,
            )
            elapsed = time.time() - t0
            ratios = _pos_neg_ratio(s)

            status_msg = {
                "computed": "計算並儲存",
                "skipped_realigned": "存在並重新對齊後儲存",
            }[status]
            print(
                f"✅ {symbol} - {label_type} 已{status_msg}"
                f"（正負比例：{ratios['pos_ratio']:.2%} / {ratios['neg_ratio']:.2%}）"
            )

            step_record = {
                "symbol": symbol,
                "label_type": label_type,
                "status": status,
                "elapsed_sec": float(elapsed),
                "nan_ratio": _nan_ratio(s),
                "index_length": int(len(s)),
                "index_match": True,
                **ratios,
            }
            run_log["steps"].append(step_record)
            update_json(paths.run_log_path(), {run_started: run_log})

            update_json(
                paths.metadata_path(),
                {
                    f"{symbol}.{label_type}": {
                        "h": HORIZON_H,
                        "state_window": STATE_WINDOW,
                        "feature_alignment": FEATURE_ALIGNMENT,
                        "saved_path": str(paths.label_path(symbol, label_type)),
                        "nan_ratio": _nan_ratio(s),
                        "index_length": int(len(s)),
                        **ratios,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                    }
                },
            )

            label_series.append(s)

        combined = pd.concat(label_series, axis=1)
        combined = combined.reindex(feature_index).sort_index()
        assert_index_matches_features(combined.index, symbol, feature_index=feature_index)

        combined_path = paths.combined_path(symbol)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(combined_path, index=True, compression="zstd", compression_level=9)
        print(f"✅ {symbol} - combined_labels（{len(combined.columns)} 欄）已儲存：{combined_path}")

        update_json(
            paths.metadata_path(),
            {
                f"{symbol}.combined": {
                    "feature_alignment": FEATURE_ALIGNMENT,
                    "index_length": int(len(combined)),
                    "index_match": True,
                    "saved_path": str(combined_path),
                    "nan_ratio": {c: float(combined[c].isna().mean()) for c in combined.columns},
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                }
            },
        )

    run_log["run_ended_utc"] = datetime.now(timezone.utc).isoformat()
    update_json(paths.run_log_path(), {run_started: run_log})
    print("\nPhase 3 標籤管線執行完畢。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3：Label Engineering 增量管線（Multi-timeframe 對齊）")
    parser.add_argument("--force", action="store_true", help="強制重算並覆寫所有 label checkpoint")
    args = parser.parse_args()
    run_label_pipeline(force_recompute=args.force)
