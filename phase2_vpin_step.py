# -*- coding: utf-8 -*-
"""
Phase 2 子任務 5（VPIN）增量計算腳本。

規範：
- 輸入：Data_Lake raw（features.feature_utils.load_raw_klines）
- 輸出：Data_Lake processed/features（features.feature_utils.save_feature）
- checkpoint：若 vpin.parquet 已存在則跳過（除非 --force）

用法：
    python phase2_vpin_step.py
    python phase2_vpin_step.py --force
"""

from __future__ import annotations

import argparse

from features.feature_utils import feature_exists, load_raw_klines, save_feature
from features.vpin import compute_vpin_features
from utils.config import SYMBOLS


def main(force_recompute: bool = False) -> int:
    for symbol in SYMBOLS:
        df = load_raw_klines(symbol)

        if force_recompute or not feature_exists(symbol, "vpin"):
            vpin_df = compute_vpin_features(df)
            p = save_feature(vpin_df, symbol, "vpin")
            print(f"✅ {symbol} - vpin（5 窗口）已計算並儲存：{p}")
        else:
            print(f"⏭️ {symbol} - vpin 已存在，跳過（如需重算請加 --force）")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 子任務 5：VPIN 增量計算")
    parser.add_argument("--force", action="store_true", help="強制重算並覆寫 vpin checkpoint")
    args = parser.parse_args()
    raise SystemExit(main(force_recompute=args.force))

