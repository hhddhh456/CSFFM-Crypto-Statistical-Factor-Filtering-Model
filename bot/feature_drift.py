# -*- coding: utf-8 -*-
"""V3：feature hash 與推論漂移監控（process 內存）。"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_prev_feature_hash: str | None = None
_prev_probas: dict[str, float] | None = None


def feature_hash(X_row: pd.DataFrame) -> str:
    """穩定 hash（欄位順序 + 數值）。"""
    cols = list(X_row.columns)
    payload = "|".join(cols).encode("utf-8") + X_row.values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def check_feature_updated(X_row: pd.DataFrame) -> dict[str, Any]:
    global _prev_feature_hash
    h = feature_hash(X_row)
    unchanged = _prev_feature_hash is not None and h == _prev_feature_hash
    if unchanged:
        logger.warning("FEATURE NOT UPDATED since last inference (hash=%s)", h)
    _prev_feature_hash = h
    return {"feature_hash": h, "unchanged_since_last": unchanged}


def log_prediction_drift(probas: dict[str, float]) -> dict[str, Any]:
    global _prev_probas
    out: dict[str, Any] = {"probas": dict(probas)}
    if _prev_probas is not None:
        diffs = {
            k: round(float(probas.get(k, 0.5)) - float(_prev_probas.get(k, 0.5)), 6)
            for k in probas
        }
        out["prediction_diff"] = diffs
        if all(abs(v) < 1e-9 for v in diffs.values()):
            logger.warning("PREDICTION UNCHANGED since last inference")
    _prev_probas = dict(probas)
    return out
