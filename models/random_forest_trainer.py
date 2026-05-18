# -*- coding: utf-8 -*-
"""
Phase 4 Random Forest 訓練（walk-forward + purge + PR-AUC）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from models.model_utils import PURGE_MINUTES, RANDOM_STATE, walk_forward_splits


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    auc: float
    accuracy: float
    pr_auc: float


@dataclass(frozen=True)
class TrainResult:
    model: RandomForestClassifier
    fold_results: list[FoldResult]
    summary: dict[str, Any]


def _safe_auc(y_true: np.ndarray, proba_pos: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba_pos))


def _safe_pr_auc(y_true: np.ndarray, proba_pos: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, proba_pos))


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    label_type: str,
    *,
    regime: str = "all_day",
    n_estimators: int = 100,
    n_splits: int = 5,
    val_size: float = 0.2,
    purge_minutes: int = PURGE_MINUTES,
    fit_final_on_full_train: bool = True,
) -> TrainResult:
    """
    訓練單一 label 的 Random Forest（walk-forward 多折）。

    若 fit_final_on_full_train=True，CV 結束後在**全部**傳入樣本上重訓最終部署模型
    （取代僅用最後一折 train 的 last_model）。
    """
    if not X.index.equals(y.index):
        raise ValueError("X 與 y 的 index 必須完全對齊")

    y_bin = y.astype("int8")
    y01 = ((y_bin == 1).astype("int8")).to_numpy()
    Xv = X.astype("float64")
    folds: list[FoldResult] = []
    last_model: RandomForestClassifier | None = None

    for fold_idx, (train_pos, val_pos) in enumerate(
        walk_forward_splits(Xv.index, n_splits=n_splits, val_size=val_size, purge_minutes=purge_minutes),
        start=1,
    ):
        X_train = Xv.iloc[train_pos]
        y_train = y01[train_pos]
        X_val = Xv.iloc[val_pos]
        y_val = y01[val_pos]

        print(
            f"[Phase4] {regime}/{label_type} fold {fold_idx}/{n_splits} "
            f"train={len(train_pos):,} val={len(val_pos):,}",
            flush=True,
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
            verbose=0,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype("int8")

        auc = _safe_auc(y_val, proba)
        pr_auc = _safe_pr_auc(y_val, proba)
        acc = float(accuracy_score(y_val, pred))

        folds.append(
            FoldResult(
                fold=fold_idx,
                n_train=int(len(train_pos)),
                n_val=int(len(val_pos)),
                auc=float(auc),
                accuracy=float(acc),
                pr_auc=float(pr_auc),
            )
        )
        last_model = model
        print(
            f"[Phase4] {regime}/{label_type} fold {fold_idx} 完成："
            f"AUC={auc:.4f} PR-AUC={pr_auc:.4f} Acc={acc:.4f}",
            flush=True,
        )

    if last_model is None:
        raise RuntimeError("無法產生任何 walk-forward fold（資料量可能不足或 purge 設定過大）")

    deploy_model = last_model
    if fit_final_on_full_train:
        deploy_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
            verbose=0,
        )
        deploy_model.fit(Xv, y01)
        print(
            f"[Phase4] {regime}/{label_type} 最終模型：全 train 重訓 n={len(Xv):,} "
            f"oob={getattr(deploy_model, 'oob_score_', float('nan')):.4f}",
            flush=True,
        )

    aucs = np.array([f.auc for f in folds], dtype="float64")
    accs = np.array([f.accuracy for f in folds], dtype="float64")
    prs = np.array([f.pr_auc for f in folds], dtype="float64")

    summary: dict[str, Any] = {
        "label_type": label_type,
        "regime": regime,
        "n_samples": int(len(Xv)),
        "n_features": int(Xv.shape[1]),
        "n_splits": int(n_splits),
        "val_size": float(val_size),
        "purge_minutes": int(purge_minutes),
        "n_estimators": int(n_estimators),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "accuracy_mean": float(np.nanmean(accs)),
        "accuracy_std": float(np.nanstd(accs)),
        "pr_auc_mean": float(np.nanmean(prs)),
        "pr_auc_std": float(np.nanstd(prs)),
        "oob_score": float(getattr(deploy_model, "oob_score_", float("nan"))),
        "fit_final_on_full_train": bool(fit_final_on_full_train),
        "cv_last_fold_oob": float(getattr(last_model, "oob_score_", float("nan"))),
    }

    return TrainResult(model=deploy_model, fold_results=folds, summary=summary)
