# -*- coding: utf-8 -*-
"""Phase 5 回測 MDA（重用 Phase 4 permutation importance）。"""

from __future__ import annotations

import pandas as pd

from models.feature_importance import compute_mda_importance, top_mda_features


def compute_backtest_mda(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    regime: str,
    *,
    n_repeats: int = 5,
) -> tuple[pd.DataFrame, list[dict]]:
    y01 = (y_test.astype("int8") == 1).astype("int8")
    mda = compute_mda_importance(
        model, X_test, y01, regime=regime, n_repeats=n_repeats, n_jobs=1
    )
    top5 = top_mda_features(mda, n=5)
    return mda, top5
