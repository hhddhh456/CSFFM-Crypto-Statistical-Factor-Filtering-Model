# -*- coding: utf-8 -*-
"""即時特徵與訓練模型特徵欄位一致性檢查。"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


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
