# -*- coding: utf-8 -*-
"""Layer A：依 OOS AUC 判斷 label 是否具推播資格。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from bot.config import OOS_AUC_MIN, OOS_COMPARISON_CSV


@lru_cache(maxsize=1)
def load_oos_table(path: str | None = None) -> pd.DataFrame:
    p = Path(path) if path else OOS_COMPARISON_CSV
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="utf-8-sig")
    if "auc_oos" not in df.columns:
        return df
    ok = df["status"] == "ok" if "status" in df.columns else pd.Series(True, index=df.index)
    return df.loc[ok].copy()


def is_label_eligible(symbol: str, regime: str, label_type: str, *, min_auc: float = OOS_AUC_MIN) -> bool:
    df = load_oos_table()
    if df.empty:
        return True
    sym = symbol.strip().upper()
    reg = regime.strip().lower()
    sub = df[(df["symbol"] == sym) & (df["regime"] == reg) & (df["label_type"] == label_type)]
    if sub.empty:
        return False
    auc = float(sub.iloc[-1]["auc_oos"])
    return auc == auc and auc >= min_auc


def get_auc_oos(symbol: str, regime: str, label_type: str) -> float:
    df = load_oos_table()
    sub = df[
        (df["symbol"] == symbol.strip().upper())
        & (df["regime"] == regime.strip().lower())
        & (df["label_type"] == label_type)
    ]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[-1]["auc_oos"])
