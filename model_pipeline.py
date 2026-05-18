# -*- coding: utf-8 -*-
"""
Phase 4 Regime-specific + Multi-timeframe 模型訓練管線。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from models.comparison import write_comparison_summary
from models.model_utils import DEFAULT_TRAIN_END_DATE, Phase4Paths, save_json_atomic, update_json
from models.regime_utils import ALL_REGIMES, regime_folder_name
from models.train_single_label import train_one


LABEL_ORDER = [
    "label_realized_volatility",
    "label_sequential_correlation",
    "label_skewness",
    "label_kurtosis",
    "label_jarque_bera",
]


def run_model_pipeline(
    force_recompute: bool = False,
    *,
    regimes: tuple[str, ...] | None = None,
    train_end_date: str = DEFAULT_TRAIN_END_DATE,
) -> None:
    p4 = Phase4Paths.default()
    run_id = datetime.now(timezone.utc).isoformat()
    regime_list = regimes or ALL_REGIMES

    run_log: dict = {
        "run_started_utc": run_id,
        "force_recompute": bool(force_recompute),
        "train_end_date": train_end_date,
        "regimes": list(regime_list),
        "steps": [],
    }
    comparison_records: list[dict] = []

    for symbol in ("BTCUSDT", "ETHUSDT"):
        print(f"\n========== {symbol} ==========")

        for regime in regime_list:
            regime_dir = p4.symbol_regime_dir(symbol, regime)
            regime_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--- {symbol} / {regime_folder_name(regime)} ---")

            all_mda_rows: list[pd.DataFrame] = []
            report: dict = {
                "symbol": symbol,
                "regime": regime,
                "run_started_utc": run_id,
                "train_end_date": train_end_date,
                "models": {},
            }

            for label_type in LABEL_ORDER:
                try:
                    res = train_one(
                        symbol,
                        label_type,
                        regime=regime,
                        train_end_date=train_end_date,
                        force_recompute=force_recompute,
                    )
                except ValueError as e:
                    print(f"⚠️ {symbol} / {regime} / {label_type} 跳過：{e}")
                    run_log["steps"].append(
                        {
                            "symbol": symbol,
                            "regime": regime,
                            "label_type": label_type,
                            "status": "skipped_error",
                            "error": str(e),
                        }
                    )
                    update_json(p4.training_log_path(), {run_id: run_log})
                    continue
                except KeyboardInterrupt:
                    run_log["steps"].append(
                        {
                            "symbol": symbol,
                            "regime": regime,
                            "label_type": label_type,
                            "status": "keyboard_interrupt",
                        }
                    )
                    update_json(p4.training_log_path(), {run_id: run_log})
                    raise

                step = {
                    "symbol": symbol,
                    "regime": regime,
                    "label_type": label_type,
                    "status": res.get("status"),
                    "model_path": res.get("model_path"),
                }

                if res.get("status") == "computed":
                    summary = res.get("summary", {})
                    top5 = res.get("top5_mda", [])
                    step.update(
                        {
                            "auc_mean": summary.get("auc_mean"),
                            "accuracy_mean": summary.get("accuracy_mean"),
                            "pr_auc_mean": summary.get("pr_auc_mean"),
                        }
                    )
                    print(
                        f"✅ {symbol} / {regime} / {label_type} | "
                        f"AUC={summary.get('auc_mean'):.4f} | "
                        f"PR-AUC={summary.get('pr_auc_mean'):.4f} | "
                        f"Acc={summary.get('accuracy_mean'):.4f}"
                    )
                    print(f"   Top5 MDA: {[t['feature'] for t in top5[:3]]}...")

                    mda: pd.DataFrame = res["mda"]
                    mda2 = mda.copy()
                    mda2.insert(0, "label_type", label_type)
                    all_mda_rows.append(mda2)

                    report["models"][label_type] = {
                        "model_path": res.get("model_path"),
                        "summary": summary,
                        "top5_mda": top5,
                        "folds": res.get("folds"),
                    }

                    comparison_records.append(
                        {
                            "symbol": symbol,
                            "regime": regime,
                            "label_type": label_type,
                            "auc_mean": summary.get("auc_mean"),
                            "accuracy_mean": summary.get("accuracy_mean"),
                            "pr_auc_mean": summary.get("pr_auc_mean"),
                            "n_samples": summary.get("n_samples"),
                            "n_features": summary.get("n_features"),
                        }
                    )
                else:
                    print(f"⏭️ {symbol} / {regime} / {label_type} 已存在，跳過")

                run_log["steps"].append(step)
                update_json(p4.training_log_path(), {run_id: run_log})

            if all_mda_rows:
                mda_all = pd.concat(all_mda_rows, axis=0, ignore_index=True)
                mda_all.to_csv(p4.regime_mda_path(symbol, regime), index=False, encoding="utf-8-sig")

            save_json_atomic(p4.regime_report_path(symbol, regime), report)

    summary_path = write_comparison_summary(comparison_records, paths=p4)
    print(f"\n✅ model_comparison_summary 已寫入：{summary_path}")

    update_json(
        p4.metadata_path(),
        {
            run_id: {
                "run_started_utc": run_id,
                "force_recompute": bool(force_recompute),
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "regimes": list(regime_list),
                "label_order": LABEL_ORDER,
                "comparison_summary": str(summary_path),
            }
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4：Regime-specific 模型訓練管線")
    parser.add_argument("--force", action="store_true", help="強制重算並覆寫所有模型檔")
    parser.add_argument(
        "--train-end",
        default=DEFAULT_TRAIN_END_DATE,
        help=f"訓練資料截止日 UTC（預設 {DEFAULT_TRAIN_END_DATE}）",
    )
    parser.add_argument(
        "--regime",
        action="append",
        dest="regimes",
        help="僅訓練指定 regime（可重複指定），預設全部",
    )
    args = parser.parse_args()
    regimes = tuple(args.regimes) if args.regimes else None
    run_model_pipeline(
        force_recompute=args.force,
        regimes=regimes,
        train_end_date=args.train_end,
    )
