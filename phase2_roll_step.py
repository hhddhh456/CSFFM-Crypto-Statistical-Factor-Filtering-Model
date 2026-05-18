# -*- coding: utf-8 -*-
"""
Phase 2 子任務 2（Roll / Roll Impact）增量計算腳本。

你目前的資料規範：
- 輸入：Data_Lake raw（由 features.feature_utils.load_raw_klines 讀）
- 輸出：Data_Lake processed/features（由 features.feature_utils.save_feature 寫）

用法：
    python phase2_roll_step.py
    python phase2_roll_step.py --force   # 忽略 checkpoint，重算並覆寫
"""

from __future__ import annotations

import argparse

from features.feature_utils import feature_exists, load_raw_klines, save_feature
from features.roll_measure import compute_roll_features, compute_roll_impact_features
from utils.config import SYMBOLS


def main(force_recompute: bool = False) -> int:
    for symbol in SYMBOLS:
        df = load_raw_klines(symbol)

        # Roll Measure
        if force_recompute or not feature_exists(symbol, "roll"):
            roll_df = compute_roll_features(df)
            p = save_feature(roll_df, symbol, "roll")
            print(f"✅ {symbol} - roll（5 窗口）已計算並儲存：{p}")
        else:
            print(f"⏭️ {symbol} - roll 已存在，跳過（如需重算請加 --force）")

        # Roll Impact
        if force_recompute or not feature_exists(symbol, "roll_impact"):
            impact_df = compute_roll_impact_features(df)
            p = save_feature(impact_df, symbol, "roll_impact")
            print(f"✅ {symbol} - roll_impact（5 窗口）已計算並儲存：{p}")
        else:
            print(f"⏭️ {symbol} - roll_impact 已存在，跳過（如需重算請加 --force）")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 子任務 2：Roll 與 Roll Impact 增量計算")
    parser.add_argument("--force", action="store_true", help="強制重算並覆寫 roll / roll_impact checkpoint")
    args = parser.parse_args()
    raise SystemExit(main(force_recompute=args.force))

