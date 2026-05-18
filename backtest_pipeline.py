# -*- coding: utf-8 -*-
"""
Phase 5 增量主控回測管線。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
import json
from datetime import datetime, timezone

from models.regime_utils import ALL_REGIMES

import pandas as pd

from backtest.diagnosis_report_writer import (
    build_diagnosis_summary,
    write_final_diagnosis_report,
    write_leakage_diagnosis_report,
    write_oos_evaluation_summary,
    write_regime_oos_comparison,
)
from backtest.diagnostics import (
    check_feature_leakage,
    check_label_leakage,
    check_time_alignment,
    summarize_leakage_risks,
)
from backtest.evaluation_utils import LABEL_ORDER, BacktestPaths, normalize_regime
from backtest.regime_comparison import write_regime_comparison
from backtest.report_writer import write_final_evaluation_report, write_settlement_focus_report
from backtest.single_label_evaluator import evaluate_out_of_sample
from backtest.single_regime_evaluator import evaluate_on_regime_slice, evaluate_regime
from backtest.window_analysis import run_window_analysis


def run_full_backtest(
    test_start_date: str = "2025-01-01",
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    regimes: tuple[str, ...] | None = None,
    label_types: tuple[str, ...] | None = None,
    force_recompute: bool = False,
) -> None:
    paths = BacktestPaths.default()
    regime_list = regimes or ALL_REGIMES
    labels = label_types or tuple(LABEL_ORDER)

    records: list[dict] = []
    cross_records: list[dict] = []

    for symbol in symbols:
        print(f"\n========== {symbol} 回測 ==========")
        for regime in regime_list:
            regime_n = normalize_regime(regime)
            print(f"\n--- {symbol} / {regime_n} ---")
            for label_type in labels:
                report_path = paths.label_report_path(symbol, regime_n, label_type)
                if report_path.is_file() and not force_recompute:
                    print(f"[SKIP] 跳過 {label_type}（已存在 report）")
                    with report_path.open("r", encoding="utf-8") as f:
                        rec = json.load(f)
                    records.append(rec)
                    continue

                try:
                    print(f"正在回測 {symbol} {regime_n} {label_type}...")
                    res = evaluate_regime(
                        symbol,
                        label_type,
                        regime_n,
                        test_start_date,
                        paths=paths,
                        save_outputs=True,
                    )
                    records.append({k: v for k, v in res.items() if k not in ("mda", "predictions")})
                    print(
                        f"[OK] {symbol}/{regime_n}/{label_type} "
                        f"AUC={res.get('auc', float('nan')):.4f} "
                        f"PR-AUC={res.get('pr_auc', float('nan')):.4f} "
                        f"Acc={res.get('accuracy', float('nan')):.4f}"
                    )
                except Exception as e:
                    print(f"[WARN] 失敗 {symbol}/{regime_n}/{label_type}: {e}")

        # 交叉：all_day 模型在 settlement 時段
        print(f"\n--- {symbol} 交叉評估：all_day 模型 @ settlement 時段 ---")
        for label_type in labels:
            try:
                cross = evaluate_on_regime_slice(
                    symbol,
                    label_type,
                    model_regime="all_day",
                    data_regime="settlement",
                    test_start_date=test_start_date,
                )
                cross_records.append(cross)
            except Exception as e:
                print(f"[WARN] 交叉評估失敗 {label_type}: {e}")

    df = write_regime_comparison(records, paths)
    write_settlement_focus_report(df, cross_records, paths)
    write_final_evaluation_report(df, paths, test_start_date=test_start_date)

    print(f"\n[OK] regime_comparison.csv：{paths.regime_comparison_path()}")
    print(f"[OK] settlement_focus_report.md：{paths.settlement_focus_path()}")
    print(f"[OK] final_evaluation_report.md：{paths.final_report_path()}")
    print(f"完成時間 UTC：{datetime.now(timezone.utc).isoformat()}")


def run_diagnostic_backtest(
    test_start_date: str = "2025-01-01",
    train_end_date: str = "2024-12-31",
    walkforward_test_start: str = "2024-01-01",
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    regimes: tuple[str, ...] | None = None,
    label_types: tuple[str, ...] | None = None,
    force_recompute: bool = False,
    include_walkforward_legacy: bool = False,
) -> None:
    """
    Phase 5 診斷管線：洩漏檢查 + 嚴格 OOS(2025+)；legacy 2024+ 對照可選。
    """
    paths = BacktestPaths.default()
    regime_list = regimes or ALL_REGIMES
    labels = label_types or tuple(LABEL_ORDER)

    leakage_checks: list[dict] = []
    oos_records: list[dict] = []

    print("\n========== Phase 5 診斷：靜態洩漏檢查 ==========")
    for symbol in symbols:
        print(f"\n--- 檢查 {symbol} ---")
        try:
            lc = check_label_leakage(symbol)
            leakage_checks.append(lc)
            print(f"  標籤檢查：risk={lc['risk_level']} passed={lc['passed']}")
        except Exception as e:
            print(f"  [WARN] 標籤檢查失敗：{e}")

        try:
            fc = check_feature_leakage(symbol)
            leakage_checks.append(fc)
            print(f"  特徵檢查：risk={fc['risk_level']} passed={fc['passed']}")
        except Exception as e:
            print(f"  [WARN] 特徵檢查失敗：{e}")

        for regime in ("all_day", "settlement"):
            try:
                ac = check_time_alignment(symbol, regime)
                leakage_checks.append(ac)
                print(f"  對齊 {regime}：risk={ac['risk_level']} n_joined={ac.get('n_joined')}")
            except Exception as e:
                print(f"  [WARN] 對齊檢查 {regime} 失敗：{e}")

    leakage_summary = summarize_leakage_risks(leakage_checks)

    total = len(symbols) * len(regime_list) * len(labels)
    done = 0

    print("\n========== Phase 5 診斷：嚴格 OOS ==========")
    if include_walkforward_legacy:
        print("（含 legacy Walk-forward 2024+ 對照）")
    for symbol in symbols:
        for regime in regime_list:
            regime_n = normalize_regime(regime)
            print(f"\n--- {symbol} / {regime_n} ---")
            for label_type in labels:
                done += 1
                oos_json = paths.oos_result_path(symbol, regime_n, label_type)
                if oos_json.is_file() and not force_recompute:
                    print(f"[SKIP] OOS {label_type}（已存在）")
                    with oos_json.open("r", encoding="utf-8") as f:
                        oos_records.append(json.load(f))
                    continue

                rec: dict = {
                    "symbol": symbol,
                    "regime": regime_n,
                    "label_type": label_type,
                }

                if include_walkforward_legacy:
                    try:
                        wf = evaluate_regime(
                            symbol,
                            label_type,
                            regime_n,
                            walkforward_test_start,
                            paths=paths,
                            save_outputs=False,
                        )
                        rec["auc_walkforward_2024plus"] = wf.get("auc", float("nan"))
                        rec["pr_auc_walkforward_2024plus"] = wf.get("pr_auc", float("nan"))
                        rec["n_walkforward"] = wf.get("n_samples", 0)
                    except Exception as e:
                        rec["auc_walkforward_2024plus"] = float("nan")
                        rec["walkforward_error"] = str(e)
                        print(f"  [WARN] Walk-forward {label_type}: {e}")

                try:
                    print(
                        f"  [{done}/{total}] 嚴格 OOS {label_type} "
                        f"(train<={train_end_date}, test>={test_start_date})..."
                    )
                    oos = evaluate_out_of_sample(
                        symbol,
                        label_type,
                        regime_n,
                        test_start=test_start_date,
                        train_end=train_end_date,
                        paths=paths,
                        save_outputs=True,
                    )
                    rec.update(oos)
                    oos_records.append(rec)
                    print(
                        f"  [OK] OOS AUC={rec.get('auc_oos', float('nan')):.4f} "
                        f"IS={rec.get('auc_in_sample', float('nan')):.4f} "
                        f"gap={rec.get('auc_gap', float('nan')):.4f} "
                        f"P4@OOS={rec.get('auc_phase4_baseline', float('nan')):.4f}"
                    )
                except Exception as e:
                    rec["status"] = "error"
                    rec["error"] = str(e)
                    oos_records.append(rec)
                    print(f"  [WARN] OOS 失敗 {label_type}: {e}")

    oos_df = pd.DataFrame(oos_records)
    oos_csv = paths.oos_comparison_path()
    oos_csv.parent.mkdir(parents=True, exist_ok=True)
    oos_df.to_csv(oos_csv, index=False, encoding="utf-8-sig")

    write_leakage_diagnosis_report(leakage_checks, leakage_summary, paths.leakage_report_path())
    write_regime_oos_comparison(oos_df, paths.regime_oos_comparison_path())

    diag_summary = build_diagnosis_summary(
        leakage_summary, oos_df, include_walkforward_legacy=include_walkforward_legacy
    )
    write_final_diagnosis_report(diag_summary, paths.final_diagnosis_path())
    write_oos_evaluation_summary(oos_df, paths.oos_evaluation_summary_path())

    print(f"\n[OK] leakage_diagnosis_report.md：{paths.leakage_report_path()}")
    print(f"[OK] regime_oos_comparison.md：{paths.regime_oos_comparison_path()}")
    print(f"[OK] oos_comparison.csv：{oos_csv}")
    print(f"[OK] final_diagnosis_report.md：{paths.final_diagnosis_path()}")
    print(f"[OK] oos_evaluation_summary.md：{paths.oos_evaluation_summary_path()}")
    print(f"完成時間 UTC：{datetime.now(timezone.utc).isoformat()}")


def run_window_analysis_pipeline(
    test_start_date: str = "2025-01-01",
    train_end_date: str = "2024-12-31",
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    regimes: tuple[str, ...] | None = None,
    label_types: tuple[str, ...] | None = None,
    force_recompute: bool = False,
) -> None:
    """多時間框架窗口：MDA 彙總 + 單窗口 OOS 消融 + 報告。"""
    from backtest.report_writer import write_final_evaluation_report

    paths = BacktestPaths.default()
    mda_df, oos_df = run_window_analysis(
        test_start=test_start_date,
        train_end=train_end_date,
        symbols=symbols,
        regimes=regimes,
        label_types=label_types,
        force_recompute=force_recompute,
        paths=paths,
    )

    regime_csv = paths.regime_comparison_path()
    if regime_csv.is_file():
        df = pd.read_csv(regime_csv, encoding="utf-8-sig")
        write_final_evaluation_report(df, paths, test_start_date=test_start_date)
        print(f"[OK] 已更新 final_evaluation_report.md（含窗口章節）")

    print(f"\n[OK] window_mda_summary.csv：{paths.window_mda_summary_path()}")
    print(f"[OK] window_oos_comparison.csv：{paths.window_oos_comparison_path()}")
    print(f"[OK] window_analysis_summary.md：{paths.window_analysis_summary_path()}")
    print(f"完成時間 UTC：{datetime.now(timezone.utc).isoformat()}")
    _ = mda_df, oos_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5：回測與診斷管線")
    parser.add_argument("--diagnostic", action="store_true", help="執行洩漏診斷與嚴格 OOS")
    parser.add_argument("--test-start", default=None, help="測試區間起始日（UTC）")
    parser.add_argument("--train-end", default="2024-12-31", help="診斷：訓練截止日（UTC）")
    parser.add_argument("--oos-start", default="2025-01-01", help="診斷：OOS 測試起始日（UTC）")
    parser.add_argument(
        "--include-walkforward-legacy",
        action="store_true",
        help="診斷時額外跑 2024+ legacy 對照（非 OOS KPI）",
    )
    parser.add_argument("--walkforward-start", default="2024-01-01", help="legacy 對照起始日")
    parser.add_argument("--symbol", action="append", dest="symbols", help="可重複指定 BTCUSDT/ETHUSDT")
    parser.add_argument("--regime", action="append", dest="regimes", help="可重複指定 regime")
    parser.add_argument("--force", action="store_true", help="強制重算並覆寫")
    parser.add_argument(
        "--window-analysis",
        action="store_true",
        help="多時間框架窗口比較（MDA 彙總 + 單窗口 OOS 消融）",
    )
    args = parser.parse_args()

    sym = tuple(args.symbols) if args.symbols else ("BTCUSDT", "ETHUSDT")
    reg = tuple(args.regimes) if args.regimes else None

    if args.window_analysis:
        test_start = args.test_start or args.oos_start or "2025-01-01"
        run_window_analysis_pipeline(
            test_start_date=test_start,
            train_end_date=args.train_end,
            symbols=sym,
            regimes=reg,
            force_recompute=args.force,
        )
    elif args.diagnostic:
        run_diagnostic_backtest(
            test_start_date=args.oos_start,
            train_end_date=args.train_end,
            walkforward_test_start=args.walkforward_start,
            symbols=sym,
            regimes=reg,
            force_recompute=args.force,
            include_walkforward_legacy=args.include_walkforward_legacy,
        )
    else:
        test_start = args.test_start or "2025-01-01"
        run_full_backtest(
            test_start_date=test_start,
            symbols=sym,
            regimes=reg,
            force_recompute=args.force,
        )
