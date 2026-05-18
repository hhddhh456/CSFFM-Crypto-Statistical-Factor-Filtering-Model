# -*- coding: utf-8 -*-
"""
Phase 4 子任務 4：Permutation Importance（MDA - Mean Decreased Accuracy）。

使用 sklearn.inspection.permutation_importance 計算特徵重要性，
並回傳 DataFrame 方便輸出 CSV。
"""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


def compute_mda_importance(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    regime: str = "all_day",
    n_repeats: int = 10,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    model
        已訓練好的 sklearn 模型（RandomForestClassifier）。
    X_val : pd.DataFrame
        驗證集特徵。
    y_val : pd.Series
        驗證集標籤（0/1）。
    n_repeats : int
        置換重複次數。

    Returns
    -------
    pd.DataFrame
        欄位：feature, importance_mean, importance_std（依重要性降序）。
    """
    res = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=n_jobs,
        scoring="accuracy",
    )
    out = pd.DataFrame(
        {
            "regime": regime,
            "feature": list(X_val.columns),
            "importance_mean": res.importances_mean,
            "importance_std": res.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    out.reset_index(drop=True, inplace=True)
    return out


def top_mda_features(mda: pd.DataFrame, n: int = 5) -> list[dict]:
    """回傳 Top N MDA 特徵（供報告使用）。"""
    top = mda.head(n)
    return [
        {
            "feature": row["feature"],
            "importance_mean": float(row["importance_mean"]),
        }
        for _, row in top.iterrows()
    ]

