# -*- coding: utf-8 -*-
"""即時特徵與訓練模型特徵欄位一致性檢查。"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def validate_feature_consistency(
    model_features: Sequence[str],
    live_features: Sequence[str],
) -> dict[str, Any]:
    """
    比對模型期望欄位與 live X_row 欄位。

    Returns:
        ok, missing_in_live, extra_in_live, vpin_enabled
    """
    model_set = {str(f) for f in model_features}
    live_set = {str(f) for f in live_features}
    missing = sorted(model_set - live_set)
    extra = sorted(live_set - model_set)
    model_vpin = {f for f in model_set if "vpin" in f.lower()}
    vpin_enabled = bool(model_vpin) and model_vpin.issubset(live_set)
    return {
        "ok": len(missing) == 0,
        "missing_in_live": missing,
        "extra_in_live": extra,
        "vpin_enabled": vpin_enabled,
    }


def get_model_feature_names(
    symbol: str,
    label_type: str,
    *,
    regime: str,
) -> list[str]:
    """從已載入 sklearn RF 取得 feature_names_in_。"""
    from models.model_utils import load_model

    model = load_model(symbol, label_type, regime=regime)
    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names) > 0:
        return [str(x) for x in names]
    n = int(getattr(model, "n_features_in_", 0) or 0)
    if n > 0:
        return [f"feature_{i}" for i in range(n)]
    return []


def align_feature_row(
    X_row: pd.DataFrame,
    model_features: Sequence[str],
) -> pd.DataFrame:
    """
    依模型訓練欄位順序對齊 X_row。
    missing → ValueError；extra → log 後 drop。
    """
    model_list = [str(f) for f in model_features]
    live_cols = list(X_row.columns)
    missing = [f for f in model_list if f not in live_cols]
    if missing:
        raise ValueError(
            f"缺少模型所需特徵欄位（{len(missing)}）：{missing[:10]}"
            + ("…" if len(missing) > 10 else "")
        )
    extra = [c for c in live_cols if c not in model_list]
    if extra:
        logger.warning("live X_row 有多餘欄位將忽略（%d）：%s", len(extra), extra[:5])

    aligned = X_row.reindex(columns=model_list).astype("float64")
    nan_cols = [c for c in model_list if aligned[c].iloc[0] != aligned[c].iloc[0]]
    if nan_cols:
        raise ValueError(f"對齊後特徵列含 NaN：{nan_cols[:10]}")
    return aligned
