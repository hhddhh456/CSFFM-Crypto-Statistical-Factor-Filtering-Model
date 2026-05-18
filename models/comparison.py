# -*- coding: utf-8 -*-
"""
Phase 4 模型表現彙總與 model_comparison_summary.md 產生。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.model_utils import Phase4Paths, save_json_atomic


def build_comparison_table(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def write_comparison_summary(
    records: list[dict],
    *,
    paths: Phase4Paths | None = None,
) -> Path:
    p4 = paths or Phase4Paths.default()
    out_path = p4.comparison_summary_path()
    df = build_comparison_table(records)

    lines = [
        "# Model Comparison Summary（Regime-specific + Multi-timeframe）",
        "",
        "跨市場特徵：50 欄（BTC/ETH 各 25 欄 Multi-timeframe）。",
        "",
    ]

    if df.empty:
        lines.append("_尚無訓練紀錄。_")
    else:
        cols = [
            "symbol",
            "regime",
            "label_type",
            "auc_mean",
            "accuracy_mean",
            "pr_auc_mean",
            "n_samples",
            "n_features",
        ]
        show = [c for c in cols if c in df.columns]
        lines.append("## 全模型表現")
        lines.append("")
        lines.append("| " + " | ".join(show) + " |")
        lines.append("| " + " | ".join(["---"] * len(show)) + " |")
        for _, row in df[show].iterrows():
            cells = [str(row[c]) if not isinstance(row[c], float) else f"{row[c]:.4f}" for c in show]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("## Settlement vs All-day（AUC 差異）")
        lines.append("")
        for sym in df["symbol"].unique():
            sub = df[df["symbol"] == sym]
            for label in sub["label_type"].unique():
                row_ad = sub[(sub["label_type"] == label) & (sub["regime"] == "all_day")]
                row_st = sub[(sub["label_type"] == label) & (sub["regime"] == "settlement")]
                if row_ad.empty or row_st.empty:
                    continue
                auc_ad = float(row_ad["auc_mean"].iloc[0])
                auc_st = float(row_st["auc_mean"].iloc[0])
                diff = auc_st - auc_ad
                lines.append(
                    f"- **{sym} / {label}**: settlement AUC={auc_st:.4f}, "
                    f"all_day AUC={auc_ad:.4f}, diff={diff:+.4f}"
                )
        lines.append("")
        lines.append("## 期權結算時段結論")
        lines.append("")
        st = df[df["regime"] == "settlement"]
        ad = df[df["regime"] == "all_day"]
        if st.empty or ad.empty:
            lines.append("資料不足，無法比較。")
        else:
            merged = st.merge(
                ad,
                on=["symbol", "label_type"],
                suffixes=("_settlement", "_all_day"),
            )
            merged["auc_diff"] = merged["auc_mean_settlement"] - merged["auc_mean_all_day"]
            n_better = int((merged["auc_diff"] > 0.02).sum())
            n_worse = int((merged["auc_diff"] < -0.02).sum())
            lines.append(
                f"- 共 {len(merged)} 組 label：settlement AUC 明顯高於 all_day（>+0.02）有 **{n_better}** 組；"
                f"明顯較低（<-0.02）有 **{n_worse}** 組。"
            )
            if n_better > n_worse:
                lines.append(
                    "- **回答：期權結算時段（UTC 06:00–08:00）部分標籤表現與全時段有明顯差異，"
                    "且部分模型在結算時段略優。**"
                )
            elif n_worse > n_better:
                lines.append(
                    "- **回答：期權結算時段表現整體不如全時段模型，顯示該時段噪音較高或樣本較少。**"
                )
            else:
                lines.append(
                    "- **回答：期權結算時段與全時段表現差異不大，未觀察到一致性的顯著優勢。**"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json_atomic(p4.models_root() / "comparison_records.json", {"records": records})
    return out_path
