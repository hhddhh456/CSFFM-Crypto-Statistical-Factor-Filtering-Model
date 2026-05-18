# -*- coding: utf-8 -*-
"""彙整回測結果為 regime_comparison.csv。"""

from __future__ import annotations

import pandas as pd

from backtest.evaluation_utils import BacktestPaths


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def add_settlement_lift(df: pd.DataFrame) -> pd.DataFrame:
    """為每個 symbol+label 計算 settlement 相對 all_day 的 AUC/PR-AUC 提升 %。"""
    if df.empty:
        return df
    rows = []
    for (sym, label), g in df.groupby(["symbol", "label_type"]):
        ad = g[g["regime"] == "all_day"]
        st = g[g["regime"] == "settlement"]
        if ad.empty or st.empty:
            continue
        auc_ad = float(ad["auc"].iloc[0])
        auc_st = float(st["auc"].iloc[0])
        pr_ad = float(ad["pr_auc"].iloc[0])
        pr_st = float(st["pr_auc"].iloc[0])
        auc_lift = ((auc_st - auc_ad) / auc_ad * 100.0) if auc_ad and auc_ad == auc_ad else float("nan")
        pr_lift = ((pr_st - pr_ad) / pr_ad * 100.0) if pr_ad and pr_ad == pr_ad else float("nan")
        rows.append(
            {
                "symbol": sym,
                "label_type": label,
                "auc_lift_pct_settlement_vs_all_day": auc_lift,
                "pr_auc_lift_pct_settlement_vs_all_day": pr_lift,
            }
        )
    lift = pd.DataFrame(rows)
    if lift.empty:
        return df
    return df.merge(lift, on=["symbol", "label_type"], how="left")


def write_regime_comparison(records: list[dict], paths: BacktestPaths | None = None) -> pd.DataFrame:
    p = paths or BacktestPaths.default()
    df = records_to_dataframe(records)
    df = add_settlement_lift(df)
    out = p.regime_comparison_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df
