# -*- coding: utf-8 -*-
"""Phase 5 診斷報告寫入。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.evaluation_utils import BacktestPaths


def write_leakage_diagnosis_report(
    checks: list[dict],
    summary: dict[str, Any],
    path: Path | None = None,
) -> Path:
    p = path or BacktestPaths.default().leakage_report_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 資料洩漏診斷報告",
        "",
        f"產生時間 UTC：{datetime.now(timezone.utc).isoformat()}",
        "",
        f"**整體風險等級：{summary.get('overall_risk', '未知')}** "
        f"（高風險項 {summary.get('n_high', 0)}，中風險項 {summary.get('n_medium', 0)}）",
        "",
    ]

    for chk in checks:
        sym = chk.get("symbol", "")
        comp = chk.get("component", "")
        regime = chk.get("regime", "")
        title = f"## {sym} / {comp}" + (f" / {regime}" if regime else "")
        lines.append(title)
        lines.append("")
        lines.append(f"- 通過：{'是' if chk.get('passed') else '否'}")
        lines.append(f"- 風險：{chk.get('risk_level', '低')}")
        if chk.get("index_start"):
            lines.append(
                f"- 索引範圍：{chk.get('index_start')} .. {chk.get('index_end')} "
                f"(joined n={chk.get('n_joined', 'N/A')})"
            )
        lines.append("")
        for f in chk.get("findings", []):
            flag = "PASS" if f.get("passed") else "FAIL"
            lines.append(
                f"- [{flag}] **{f.get('check', '')}** ({f.get('risk_level', '')})：{f.get('detail', '')}"
            )
        lines.append("")

    table: pd.DataFrame = summary.get("table", pd.DataFrame())
    if len(table) > 0:
        lines.append("## 檢查彙總表")
        lines.append("")
        cols = ["symbol", "regime", "component", "check", "risk_level", "passed", "detail"]
        cols = [c for c in cols if c in table.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in table.iterrows():
            cells = [str(row[c]) for c in cols]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 風險判定規則（參考）")
    lines.append("")
    lines.append("| 環節 | 高風險條件 | 建議 |")
    lines.append("|------|------------|------|")
    lines.append("| Phase 4 全期訓練 | OOS AUC 遠低於 Phase4 baseline | 重訓並限定 train<=2024 |")
    lines.append("| 標籤 shift(-h) | 重算一致率 <99% | 修 Phase 3 |")
    lines.append("| 特徵 | ffill 導致樣本異常膨脹 | 訓練/回測用 dropna |")
    lines.append("| Purge | HORIZON_H ≠ PURGE_MINUTES | 已設 1500，維持一致 |")
    lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_regime_oos_comparison(df: pd.DataFrame, path: Path | None = None) -> Path:
    p = path or BacktestPaths.default().regime_oos_comparison_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Regime Out-of-Sample 比較（2025+）",
        "",
        f"產生時間 UTC：{datetime.now(timezone.utc).isoformat()}",
        "",
        "欄位說明：`auc_in_sample` = walk-forward CV 均值；`auc_oos` = 2025+ 嚴格 OOS；"
        "`pr_auc_walkforward_2024plus` 僅 legacy 對照（2024 在 train 內）。",
        "",
    ]

    if df.empty:
        lines.append("_尚無 OOS 評估資料。_")
    else:
        show = [
            c
            for c in [
                "symbol",
                "regime",
                "label_type",
                "n_oos",
                "auc_in_sample",
                "auc_holdout_train",
                "auc_oos",
                "auc_gap",
                "pr_auc_walkforward_2024plus",
                "auc_phase4_baseline",
                "baseline_minus_fresh_oos",
                "status",
            ]
            if c in df.columns
        ]
        lines.append("| " + " | ".join(show) + " |")
        lines.append("| " + " | ".join(["---"] * len(show)) + " |")
        for _, row in df.iterrows():
            cells = []
            for c in show:
                v = row[c]
                if isinstance(v, float):
                    cells.append(f"{v:.4f}" if v == v else "nan")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        # Settlement vs all_day OOS 均值
        ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df
        if len(ok) and "auc_oos" in ok.columns:
            lines.append("## Settlement vs All Day（OOS AUC 均值）")
            lines.append("")
            for sym in ok["symbol"].unique():
                sub = ok[ok["symbol"] == sym]
                ad = sub[sub["regime"] == "all_day"]["auc_oos"].mean()
                st = sub[sub["regime"] == "settlement"]["auc_oos"].mean()
                lines.append(f"- **{sym}** all_day OOS AUC 均值={ad:.4f}；settlement={st:.4f}")
                if ad == ad and st == st:
                    lines.append(f"  - settlement - all_day = {st - ad:+.4f}")
            lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_final_diagnosis_report(
    summary: dict[str, Any],
    path: Path | None = None,
) -> Path:
    p = path or BacktestPaths.default().final_diagnosis_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Phase 5 最終診斷報告",
        "",
        f"產生時間 UTC：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. 目前 AUC 0.90 是否存在明顯資料洩漏？",
        "",
        summary.get("q1_leakage", "_待填入_"),
        "",
        "## 2. Out-of-Sample（2025 年後）表現如何？",
        "",
        summary.get("q2_oos", "_待填入_"),
        "",
        "## 3. Settlement 模型在真實未來資料上是否真的比較好？",
        "",
        summary.get("q3_settlement", "_待填入_"),
        "",
        "## 4. 下一步應該怎麼修正？",
        "",
        summary.get("q4_next_steps", "_待填入_"),
        "",
        "---",
        "",
        "### 數據摘要",
        "",
    ]

    for k, v in summary.get("stats", {}).items():
        lines.append(f"- **{k}**：{v}")

    lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_oos_evaluation_summary(df: pd.DataFrame, path: Path | None = None) -> Path:
    """由 oos_comparison.csv 產生 OOS 主 KPI 摘要（決策用）。"""
    p = path or BacktestPaths.default().oos_evaluation_summary_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# OOS 評估摘要（主 KPI）",
        "",
        f"產生時間 UTC：{datetime.now(timezone.utc).isoformat()}",
        "",
        "> **決策請以 `auc_oos` / `pr_auc_oos` 為準。** "
        "`pr_auc_walkforward_2024plus` 含 2024 in-train 區間，僅供 legacy 對照。",
        "",
    ]

    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    if ok.empty:
        lines.append("_尚無 OOS 資料。_")
    else:
        if "auc_oos" in ok.columns:
            summary = (
                ok.groupby(["symbol", "regime"])[["auc_oos", "pr_auc_oos", "auc_in_sample", "auc_gap"]]
                .mean()
                .reset_index()
            )
            cols = list(summary.columns)
            lines.append("## 各 Symbol × Regime 均值（OOS 主表）")
            lines.append("")
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for _, row in summary.iterrows():
                cells = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

        lines.append("## 決策門檻（參考）")
        lines.append("")
        mean_oos = float(ok["auc_oos"].mean())
        lines.append(f"- 全體 OOS AUC 均值：**{mean_oos:.4f}**")
        if mean_oos >= 0.55:
            lines.append("- 達通過線（>0.55）：可進入小規模策略回測，但仍需單 label 檢視")
        elif mean_oos < 0.52:
            lines.append("- 低於暫停線（<0.52）：建議優先改標籤/特徵或暫停專案")
        else:
            lines.append("- 介於 0.52–0.55：訊號偏弱，謹慎推進")

        if "regime" in ok.columns:
            lines.append("")
            lines.append("## Settlement vs All Day（OOS）")
            lines.append("")
            for sym in ok["symbol"].unique():
                sub = ok[ok["symbol"] == sym]
                ad = sub[sub["regime"] == "all_day"]["auc_oos"].mean()
                st = sub[sub["regime"] == "settlement"]["auc_oos"].mean()
                better_count = 0
                for lbl in sub["label_type"].unique():
                    a = sub[(sub["regime"] == "all_day") & (sub["label_type"] == lbl)]["auc_oos"]
                    s = sub[(sub["regime"] == "settlement") & (sub["label_type"] == lbl)]["auc_oos"]
                    if len(a) and len(s) and float(s.iloc[0]) > float(a.iloc[0]):
                        better_count += 1
                n_lbl = len(sub["label_type"].unique())
                lines.append(
                    f"- **{sym}**：all_day={ad:.4f}，settlement={st:.4f}；"
                    f"settlement 較優 label 數={better_count}/{n_lbl}"
                )

    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def build_diagnosis_summary(
    leakage_summary: dict[str, Any],
    oos_df: pd.DataFrame,
    *,
    include_walkforward_legacy: bool = False,
) -> dict[str, Any]:
    """依 OOS 與洩漏檢查結果產生四問答案。"""
    overall = leakage_summary.get("overall_risk", "低")
    n_high = leakage_summary.get("n_high", 0)

    ok = oos_df[oos_df["status"] == "ok"] if "status" in oos_df.columns else oos_df
    mean_oos = float(ok["auc_oos"].mean()) if len(ok) and "auc_oos" in ok.columns else float("nan")
    mean_cv = float(ok["auc_in_sample"].mean()) if len(ok) and "auc_in_sample" in ok.columns else float("nan")
    mean_gap = float(ok["auc_gap"].mean()) if len(ok) and "auc_gap" in ok.columns else float("nan")
    mean_p4 = (
        float(ok["auc_phase4_baseline"].mean())
        if len(ok) and "auc_phase4_baseline" in ok.columns
        else float("nan")
    )
    mean_baseline_diff = (
        float(ok["baseline_minus_fresh_oos"].mean())
        if len(ok) and "baseline_minus_fresh_oos" in ok.columns
        else float("nan")
    )
    wf_mean = (
        float(ok["pr_auc_walkforward_2024plus"].mean())
        if len(ok) and "pr_auc_walkforward_2024plus" in ok.columns
        else float("nan")
    )

    q1 = f"靜態檢查整體風險為 **{overall}**（高風險子項 {n_high} 個）。"
    if n_high > 0:
        q1 += " 存在高風險靜態檢查項，請見 leakage_diagnosis_report.md。"
    elif mean_baseline_diff == mean_baseline_diff and mean_baseline_diff > 0.10:
        q1 += (
            f" Phase4 與 OOS 部署模型差距 {mean_baseline_diff:.3f}，"
            "**建議確認 Phase 4 已限定 train<=2024 並重訓**。"
        )
    elif mean_p4 == mean_p4 and mean_oos == mean_oos and abs(mean_p4 - mean_oos) < 0.02:
        q1 += (
            " Phase4 與嚴格 OOS 模型在同一 2025+ 切片表現一致，"
            "**舊版 0.90 虛高主要來自已修正之評估切分，非標籤 shift 錯誤**。"
        )
    elif mean_gap == mean_gap and mean_gap > 0.15:
        q1 += f" CV 與 OOS 平均差距 {mean_gap:.3f}，泛化有限但**不等同資料洩漏**。"
    else:
        q1 += " 未觀察到 Phase4/OOS 不一致或極端靜態風險。"

    q2 = (
        f"嚴格 OOS（2025+）平均 AUC = **{mean_oos:.4f}**（CV 均值 {mean_cv:.4f}，gap {mean_gap:.4f}，n={len(ok)}）。"
        if mean_oos == mean_oos
        else "無足夠 2025+ 樣本完成 OOS 評估。"
    )
    if mean_p4 == mean_p4 and mean_oos == mean_oos:
        q2 += f" Phase4 部署模型 @ OOS：**{mean_p4:.4f}**。"
    if include_walkforward_legacy and wf_mean == wf_mean:
        q2 += f" Legacy 2024+ PR-AUC 均值 **{wf_mean:.4f}**（勿當 OOS KPI）。"

    q3_parts = []
    if len(ok) and "regime" in ok.columns:
        for sym in ok["symbol"].unique():
            sub = ok[ok["symbol"] == sym]
            ad = sub[sub["regime"] == "all_day"]["auc_oos"].mean()
            st = sub[sub["regime"] == "settlement"]["auc_oos"].mean()
            if ad == ad and st == st:
                better = "是" if st > ad else "否"
                q3_parts.append(
                    f"{sym}：settlement OOS={st:.4f} vs all_day={ad:.4f}，settlement 較優={better}"
                )
    q3 = "；".join(q3_parts) if q3_parts else "資料不足。"

    steps: list[str] = []
    if overall in ("高", "中"):
        steps.append("1. 依 leakage_diagnosis_report.md 完成 Data 清理（dropna、移除 smoke parquet）")
    if mean_oos == mean_oos and mean_oos < 0.55:
        steps.append(
            f"{len(steps)+1}. OOS AUC 均值 {mean_oos:.2f}<0.55：評估改標籤/特徵/模型或暫停專案"
        )
    elif mean_oos == mean_oos and mean_oos >= 0.55:
        steps.append(
            f"{len(steps)+1}. OOS 達弱通過線：以 oos_evaluation_summary.md 挑選 label 做小規模策略回測"
        )
    steps.append(f"{len(steps)+1}. 主回測使用 `python backtest_pipeline.py --test-start 2025-01-01`")
    if not steps:
        steps.append("1. 維持現有切分，定期重跑 --diagnostic 監控 OOS 漂移")
    q4 = "\n".join(steps)

    stats = {
        "leakage_overall_risk": overall,
        "oos_evaluations_ok": len(ok),
        "mean_auc_cv": f"{mean_cv:.4f}" if mean_cv == mean_cv else "nan",
        "mean_auc_oos": f"{mean_oos:.4f}" if mean_oos == mean_oos else "nan",
        "mean_auc_gap_cv_minus_oos": f"{mean_gap:.4f}" if mean_gap == mean_gap else "nan",
        "mean_phase4_baseline_oos": f"{mean_p4:.4f}" if mean_p4 == mean_p4 else "nan",
    }
    if wf_mean == wf_mean:
        stats["walkforward_2024plus_pr_auc_mean_legacy"] = f"{wf_mean:.4f}"

    return {
        "q1_leakage": q1,
        "q2_oos": q2,
        "q3_settlement": q3,
        "q4_next_steps": q4,
        "stats": stats,
    }
