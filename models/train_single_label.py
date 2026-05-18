# -*- coding: utf-8 -*-
"""
Phase 4 單一標籤 × regime 訓練入口。

訓練資料預設截止 2024-12-31（UTC），2025+ 保留給 Phase 5 嚴格 OOS。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from models.feature_importance import compute_mda_importance, top_mda_features
from models.feature_preparation import prepare_cross_market_features
from models.model_utils import (
    DEFAULT_TRAIN_END_DATE,
    Phase4Paths,
    mask_index_up_to_train_end,
    save_model,
)
from models.random_forest_trainer import train_random_forest


def _select_label(y_all: pd.DataFrame, label_type: str) -> pd.Series:
    if label_type not in y_all.columns:
        raise ValueError(f"找不到標籤欄位：{label_type}，可用欄位：{list(y_all.columns)}")
    return y_all[label_type]


def train_one(
    symbol: str,
    label_type: str,
    *,
    regime: str = "all_day",
    train_end_date: str = DEFAULT_TRAIN_END_DATE,
    force_recompute: bool = False,
) -> dict:
    p4 = Phase4Paths.default()
    model_path = p4.model_path(symbol, regime, label_type)
    if model_path.is_file() and not force_recompute:
        return {"status": "skipped", "model_path": str(model_path), "regime": regime}

    prepared = prepare_cross_market_features(symbol, regime=regime)
    X = prepared.X
    y = _select_label(prepared.y_all, label_type)

    mask = ~(X.isna().any(axis=1) | y.isna())
    X2 = X.loc[mask]
    y2 = y.loc[mask]

    train_mask = mask_index_up_to_train_end(X2.index, train_end_date)
    X_train = X2.loc[train_mask]
    y_train = y2.loc[train_mask]

    if len(X_train) < 5000:
        raise ValueError(
            f"{symbol} {regime} {label_type} 在 train_end<={train_end_date} 後樣本過少：{len(X_train)}"
        )

    print(
        f"[Phase4] {symbol}/{regime}/{label_type} 訓練樣本 "
        f"n={len(X_train):,}（截止 {train_end_date}），排除 2025+ {int((~train_mask).sum()):,} 列",
        flush=True,
    )

    result = train_random_forest(
        X_train,
        y_train,
        label_type,
        regime=regime,
        fit_final_on_full_train=True,
    )
    p = save_model(result.model, symbol, label_type, regime=regime, paths=p4)

    cut = int(len(X_train) * 0.8)
    X_val = X_train.iloc[cut:]
    y_val01 = (y_train.iloc[cut:].astype("int8") == 1).astype("int8")
    mda = compute_mda_importance(result.model, X_val, y_val01, regime=regime, n_repeats=10)
    top5 = top_mda_features(mda, n=5)

    summary = dict(result.summary)
    summary["train_end_date"] = train_end_date
    summary["n_samples_total_after_mask"] = int(len(X2))
    summary["n_samples_excluded_after_train_end"] = int((~train_mask).sum())

    return {
        "status": "computed",
        "model_path": str(p),
        "regime": regime,
        "train_end_date": train_end_date,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "folds": [f.__dict__ for f in result.fold_results],
        "mda": mda,
        "top5_mda": top5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4：訓練單一 label × regime")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--regime", default="all_day")
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END_DATE, help="訓練截止日 UTC")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    res = train_one(
        args.symbol,
        args.label,
        regime=args.regime,
        train_end_date=args.train_end,
        force_recompute=args.force,
    )
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
