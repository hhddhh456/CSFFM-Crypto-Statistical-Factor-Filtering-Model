# -*- coding: utf-8 -*-
"""
Phase 1 主控腳本：依序下載 BTCUSDT / ETHUSDT 1 分鐘 K 線、清洗並寫入 Data_Lake。

執行前請確認：
- 已設定環境變數 DATA_LAKE_ROOT（若本機無 D 槽，請指向有效路徑，例如 C:\\Data_Lake）
- 已安裝 requirements.txt

用法：
    python data_pipeline.py
    python data_pipeline.py --force   # 強制重新下載（清除該幣種 checkpoint）
"""

from __future__ import annotations

import argparse
import sys

from utils.binance_downloader import download_binance_klines
from utils.config import END_DATE, PARQUET_DATE_RANGE_LABEL, START_DATE, SYMBOLS, ensure_raw_crypto_dir, final_parquet_path
from utils.data_validator import clean_klines, validate_klines


def main(force_redownload: bool = False) -> int:
    """
    下載、清洗、覆寫寫入正式 Parquet，並列印驗證摘要。

    Returns:
        0 表示成功；1 表示發生錯誤。
    """
    try:
        ensure_raw_crypto_dir()
    except OSError as e:
        print(
            f"無法建立 Data_Lake 目錄。\n錯誤：{e}\n"
            f"建議：檢查 DATA_LAKE_ROOT 是否指向有效磁碟，或以系統管理員身分重試。"
        )
        return 1

    for symbol in SYMBOLS:
        try:
            df = download_binance_klines(
                symbol,
                START_DATE,
                END_DATE,
                force_redownload=force_redownload,
            )
            cleaned = clean_klines(df, symbol)
            out_path = final_parquet_path(symbol, "1m", date_range_label=PARQUET_DATE_RANGE_LABEL)
            cleaned.to_parquet(out_path, index=True)
        except Exception as e:
            print(f"處理 {symbol} 時發生錯誤：\n{e}")
            return 1

        report = validate_klines(symbol, date_range=PARQUET_DATE_RANGE_LABEL)
        n = len(cleaned)
        print(
            f"✅ {out_path.name} 下載並清洗完成，共 {n:,} 筆資料\n"
            f"   時間範圍（UTC）：{cleaned.index.min()} ~ {cleaned.index.max()}\n"
            f"   驗證：重複時間戳 {report.get('重複時間戳筆數', '—')} 筆；"
            f"收盤價缺失率 {report.get('收盤價缺失率', '—')}"
        )

    print("\nPhase 1 管線執行完畢。後續請使用 load_raw_klines(symbol, \"1m\", \"2021-2026\") 讀取。")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1：Binance 1m K 線下載與清洗")
    parser.add_argument(
        "--force",
        action="store_true",
        help="強制重新下載並清除各幣種正式工作階段之 checkpoint／log 條目",
    )
    args = parser.parse_args()
    sys.exit(main(force_redownload=args.force))
