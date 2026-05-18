# -*- coding: utf-8 -*-
"""
Phase 5 單一 label 嚴格 Out-of-Sample 評估（訓練 <= train_end，測試 >= test_start）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from backtest.evaluation_utils import (
    BacktestPaths,
    classification_metrics,
    labels_to_binary,
    load_prepared_dataset,
    load_regime_model,
    split_xy_by_label,
    train_oos_masks,
)
from models.model_utils import RANDOM_STATE
from models.random_forest_trainer import train_random_forest


def _compute_holdout_train_auc(
    X_train,
    y_train,
    *,
    n_estimators: int = 100,
) -> float:
    """train 前 80% fit、後 20% eval（與部署用全 train fit 分開）。"""
    cut = int(len(X_train) * 0.8)
    if cut < 500 or len(X_train) - cut < 200:
        return float("nan")
    y_fit = labels_to_binary(y_train.iloc[:cut])
    y_val = labels_to_binary(y_train.iloc[cut:])
    if len(np.unique(y_fit)) < 2 or len(np.unique(y_val)) < 2:
        return float("nan")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        bootstrap=True,
    )
    clf.fit(X_train.iloc[:cut], y_fit)
    proba = clf.predict_proba(X_train.iloc[cut:])[:, 1]
    pred = (proba >= 0.5).astype(np.int8)
    return classification_metrics(y_val, pred, proba).get("auc", float("nan"))


def evaluate_out_of_sample(
    symbol: str,
    label_type: str,
    regime: str,
    test_start: str = "2025-01-01",
    train_end: str = "2024-12-31",
    *,
    paths: Optional[BacktestPaths] = None,
    save_outputs: bool = True,
) -> dict:
    """
    嚴格 OOS：在 train_end 以前重新訓練 RF，在 test_start 以後評估。
    auc_in_sample = walk-forward CV 均值；auc_gap = CV 均值 - OOS AUC。
    """
    p = paths or BacktestPaths.default()
    sym = symbol.strip().upper()
    regime_n = regime.strip().lower()

    X, y = split_xy_by_label(load_prepared_dataset(sym, regime_n), label_type)
    train_mask, oos_mask = train_oos_masks(X.index, train_end=train_end, test_start=test_start)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_oos = X.loc[oos_mask]
    y_oos = y.loc[oos_mask]

    report: dict = {
        "symbol": sym,
        "label_type": label_type,
        "regime": regime_n,
        "train_end": train_end,
        "test_start": test_start,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(X_train)),
        "n_oos": int(len(X_oos)),
        "auc_in_sample": float("nan"),
        "auc_holdout_train": float("nan"),
        "auc_oos": float("nan"),
        "auc_gap": float("nan"),
        "pr_auc_oos": float("nan"),
        "auc_phase4_baseline": float("nan"),
        "status": "pending",
    }

    if len(X_train) < 5000:
        report["status"] = "insufficient_train"
        return report

    train_result = train_random_forest(X_train, y_train, label_type, regime=regime_n)
    model = train_result.model
    auc_cv = float(train_result.summary.get("auc_mean", float("nan")))
    report["walk_forward_auc_mean_train"] = auc_cv
    report["auc_in_sample"] = auc_cv
    report["pr_auc_in_sample_cv"] = float(train_result.summary.get("pr_auc_mean", float("nan")))
    report["auc_holdout_train"] = _compute_holdout_train_auc(X_train, y_train)

    if len(X_oos) == 0:
        report["status"] = "no_oos_samples"
        return report

    y_oos01 = labels_to_binary(y_oos)
    proba_oos = model.predict_proba(X_oos)[:, 1]
    pred_oos = (proba_oos >= 0.5).astype(np.int8)
    m_oos = classification_metrics(y_oos01, pred_oos, proba_oos)

    auc_oos = m_oos.get("auc", float("nan"))
    report["auc_oos"] = auc_oos
    report["pr_auc_oos"] = m_oos.get("pr_auc", float("nan"))
    report["accuracy_oos"] = m_oos.get("accuracy", float("nan"))
    report["f1_oos"] = m_oos.get("f1", float("nan"))
    if auc_cv == auc_cv and auc_oos == auc_oos:
        report["auc_gap"] = float(auc_cv - auc_oos)
    report["status"] = "ok"

    try:
        p4_model = load_regime_model(sym, label_type, regime_n)
        proba_p4 = p4_model.predict_proba(X_oos)[:, 1]
        pred_p4 = (proba_p4 >= 0.5).astype(np.int8)
        m_p4 = classification_metrics(y_oos01, pred_p4, proba_p4)
        report["auc_phase4_baseline"] = m_p4.get("auc", float("nan"))
        report["pr_auc_phase4_baseline"] = m_p4.get("pr_auc", float("nan"))
        if report["auc_phase4_baseline"] == report["auc_phase4_baseline"] and auc_oos == auc_oos:
            report["baseline_minus_fresh_oos"] = float(report["auc_phase4_baseline"] - auc_oos)
    except Exception as e:
        report["auc_phase4_baseline_error"] = str(e)

    if save_outputs:
        out_path = p.oos_result_path(sym, regime_n, label_type)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in report.items() if not isinstance(v, (np.ndarray,))}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
        report["oos_json_path"] = str(out_path)

    return report
