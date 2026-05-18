# -*- coding: utf-8 -*-
"""Phase 5 settlement 專項與最終總結報告。"""

from __future__ import annotations

import pandas as pd

from backtest.evaluation_utils import BacktestPaths
from backtest.window_analysis import _rank_windows_by_mda, _rank_windows_by_oos


def _pct_lift(new: float, base: float) -> str:
    if base != base or new != new or base == 0:
        return "N/A"
    return f"{(new - base) / base * 100:+.2f}%"


def write_settlement_focus_report(
    df: pd.DataFrame,
    cross_records: list[dict],
    paths: BacktestPaths | None = None,
) -> None:
    p = paths or BacktestPaths.default()
    out = p.settlement_focus_path()

    lines = [
        "# Settlement 時段專項分析（UTC 06:00–08:00）",
        "",
        "本報告比較 **settlement 專用模型** 與 **全時段模型在 settlement 切片上** 的表現。",
        "",
    ]

    st = df[df["regime"] == "settlement"].copy()
    if st.empty:
        lines.append("_尚無 settlement 回測資料。_")
    else:
        lines.append("## Settlement 模型（訓練＋測試皆在 settlement 時段）")
        lines.append("")
        cols = ["symbol", "label_type", "auc", "pr_auc", "accuracy", "f1", "n_samples", "strategy_return", "win_rate"]
        cols = [c for c in cols if c in st.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in st.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if cross_records:
        lines.append("## 全時段模型 @ Settlement 時段（交叉評估）")
        lines.append("")
        cross_df = pd.DataFrame(cross_records)
        for _, row in cross_df.iterrows():
            lines.append(
                f"- **{row['symbol']} / {row['label_type']}**: "
                f"AUC={row.get('auc', float('nan')):.4f}, PR-AUC={row.get('pr_auc', float('nan')):.4f}, "
                f"策略報酬={row.get('strategy_return', 0):.2f}"
            )
        lines.append("")

    lines.append("## 結論（Settlement vs All-day 模型）")
    lines.append("")
    for sym in df["symbol"].unique() if not df.empty else []:
        for label in df["label_type"].unique() if not df.empty else []:
            sub = df[(df["symbol"] == sym) & (df["label_type"] == label)]
            ad = sub[sub["regime"] == "all_day"]
            st_row = sub[sub["regime"] == "settlement"]
            if ad.empty or st_row.empty:
                continue
            lines.append(
                f"- {sym} / {label}: settlement 模型 AUC={st_row['auc'].iloc[0]:.4f} vs "
                f"all_day 模型 AUC={ad['auc'].iloc[0]:.4f} "
                f"（提升 {_pct_lift(st_row['auc'].iloc[0], ad['auc'].iloc[0])}）"
            )

    lines.append("")
    lines.append("## 交易建議（簡易策略，含手續費）")
    lines.append("")
    lines.append("- 僅在模型機率 ≥ 0.55（或 ≤ 0.45）時視為有訊號；預測正確 +1、錯誤 -1，每筆扣雙邊手續費。")
    lines.append("- Settlement 樣本較少，指標波動大，建議以 PR-AUC 與樣本數一併判斷。")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_evaluation_report(
    df: pd.DataFrame,
    paths: BacktestPaths | None = None,
    *,
    test_start_date: str = "2025-01-01",
) -> None:
    p = paths or BacktestPaths.default()
    out = p.final_report_path()

    lines = [
        "# Phase 5 最終回測評估報告",
        "",
        f"> **測試區間起始：{test_start_date} UTC 之後。** "
        "決策請優先參考 `oos_evaluation_summary.md` 與 `oos_comparison.csv`（嚴格 OOS）。",
        "",
        "特徵：50 維 Multi-timeframe（btc_/eth_ 各 25 欄）。",
        "模型：Phase 4 四種 regime（train<=2024-12-31）。",
        "",
        "SHAP：未啟用（可選擴充）。",
        "",
        "## 各 Regime 模型表現總覽（本報告區間）",
        "",
    ]

    if df.empty:
        lines.append("_無資料。_")
    else:
        summary = (
            df.groupby(["symbol", "regime"])[["auc", "pr_auc", "accuracy", "f1", "strategy_return"]]
            .mean()
            .reset_index()
        )
        cols = list(summary.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in summary.iterrows():
            cells = [str(row[c]) if not isinstance(row[c], float) else f"{row[c]:.4f}" for c in cols]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        if "auc_lift_pct_settlement_vs_all_day" in df.columns:
            lines.append("## Settlement 相較全時段提升（同 label 平均）")
            lines.append("")
            lift = df.groupby("symbol")[
                ["auc_lift_pct_settlement_vs_all_day", "pr_auc_lift_pct_settlement_vs_all_day"]
            ].mean()
            lines.append(lift.to_string())
            lines.append("")

        lines.append("## Top 5 MDA 特徵（依 regime 彙整）")
        lines.append("")
        lines.append("詳見各目錄 `mda_label_*.csv` 與 `report_label_*.json` 的 top5_mda 欄位。")
        lines.append("")

        lines.append("## 模擬交易價值（扣費後）")
        lines.append("")
        by_regime = df.groupby("regime")["strategy_return"].sum()
        for reg, val in by_regime.items():
            lines.append(f"- **{reg}** 累積策略報酬（全 label 加總）：{val:.2f}")
        lines.append("")

        lines.append("## 期權結算時段是否明顯不同？")
        lines.append("")
        if "auc_lift_pct_settlement_vs_all_day" in df.columns:
            mean_lift = df["auc_lift_pct_settlement_vs_all_day"].mean()
            lines.append(
                f"- 平均 AUC 提升（settlement 模型 vs all_day 模型，同切片訓練邏輯）：**{mean_lift:+.2f}%**"
            )
            if mean_lift > 2:
                lines.append("- **回答：部分標籤在 settlement 時段模型優於全時段，值得優先使用 settlement 專用模型。**")
            elif mean_lift < -2:
                lines.append("- **回答：全時段模型整體仍較穩，settlement 專用模型未顯著更好。**")
            else:
                lines.append("- **回答：差異不大，需結合單一 label 與 PR-AUC 再決策。**")
        else:
            lines.append("- 資料不足，無法判斷。")

    lines.extend(_window_analysis_section(p))

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _window_analysis_section(p: BacktestPaths) -> list[str]:
    """從 window_* CSV 產出最終報告中的多時間框架章節。"""
    oos_path = p.window_oos_comparison_path()
    mda_path = p.window_mda_summary_path()
    summary_path = p.window_analysis_summary_path()

    if not oos_path.is_file() and not mda_path.is_file():
        return [
            "",
            "## 多時間框架窗口（50–720 分鐘）",
            "",
            "_尚未執行窗口分析。請執行：_",
            "`python backtest_pipeline.py --window-analysis --test-start 2025-01-01`",
            "",
        ]

    lines = [
        "",
        "## 多時間框架窗口（50–720 分鐘）",
        "",
        "特徵回顧窗口與 UTC **regime** 為不同維度；本節比較 Phase 2 五種窗口。",
        f"完整說明見 [`window_analysis_summary.md`]({summary_path.name})。",
        "",
    ]

    if oos_path.is_file():
        oos_df = pd.read_csv(oos_path, encoding="utf-8-sig")
        oos_rank = _rank_windows_by_oos(oos_df)
        if not oos_rank.empty:
            best_w = int(oos_rank.iloc[0]["window"])
            lines.append(
                f"- **OOS 消融（主 KPI）**：最佳窗口 **{best_w} 分鐘**"
                f"（平均 AUC {oos_rank.iloc[0]['mean_auc_oos']:.4f}）；"
                f"最弱 **{int(oos_rank.iloc[-1]['window'])} 分鐘**"
                f"（{oos_rank.iloc[-1]['mean_auc_oos']:.4f}）。"
            )
            lines.append(f"- 細項：`{oos_path.name}`（{len(oos_df)} 列）")
        else:
            lines.append("- OOS 窗口消融：尚無有效結果。")

    if mda_path.is_file():
        mda_df = pd.read_csv(mda_path, encoding="utf-8-sig")
        mda_rank = _rank_windows_by_mda(mda_df)
        if not mda_rank.empty:
            col = "importance_mean_avg" if "importance_mean_avg" in mda_rank.columns else "importance_mean_sum"
            mda_best = int(mda_rank.iloc[0]["window"])
            lines.append(
                f"- **MDA 彙總（輔助）**：重要性最高窗口 **{mda_best} 分鐘**"
                f"（{col}={mda_rank.iloc[0][col]:.6f}）。"
            )
            lines.append(f"- 細項：`{mda_path.name}`")

    lines.append("")
    return lines
