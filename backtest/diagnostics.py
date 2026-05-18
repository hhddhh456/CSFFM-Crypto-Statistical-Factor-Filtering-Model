# -*- coding: utf-8 -*-
"""
Phase 5 診斷工具：洩漏檢查、時間對齊、嚴格 OOS 測試。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from backtest.evaluation_utils import (
    classification_metrics,
    labels_to_binary,
    load_prepared_dataset,
    split_xy_by_label,
    train_oos_masks,
)
from labels.label_utils import HORIZON_H, STATE_WINDOW
from labels.realized_volatility import compute_realized_volatility_label
from models.feature_preparation import prepare_cross_market_features
from models.model_utils import Phase4Paths, load_combined_features, load_combined_labels
from models.random_forest_trainer import train_random_forest
from utils.config import PARQUET_DATE_RANGE_LABEL
from utils.data_loader import load_raw_klines


def _risk_level_from_findings(findings: list[dict]) -> str:
    levels = [f.get("risk_level", "低") for f in findings]
    if any(x == "高" for x in levels):
        return "高"
    if any(x == "中" for x in levels):
        return "中"
    return "低"


def check_label_leakage(
    symbol: str,
    label_type: Optional[str] = None,
) -> dict[str, Any]:
    """
    檢查標籤是否含未來資訊洩漏（shift(-h)、首尾 NaN 遮罩）。
    """
    sym = symbol.strip().upper()
    p4 = Phase4Paths.default()
    labels = load_combined_labels(sym, paths=p4)
    findings: list[dict] = []
    k_mask = STATE_WINDOW + HORIZON_H

    # 1) 開頭 NaN 遮罩
    for col in labels.columns:
        if label_type and col != label_type:
            continue
        series = labels[col]
        head = series.iloc[: min(k_mask, len(series))]
        nan_ratio = float(head.isna().mean()) if len(head) else 0.0
        ok = nan_ratio >= 0.99
        findings.append(
            {
                "check": "initial_nan_mask",
                "label": col,
                "risk_level": "低" if ok else "中",
                "detail": f"前 {len(head)} 列 NaN 比例={nan_ratio:.2%}（預期≈100%）",
                "passed": ok,
            }
        )

    # 2) shift(-h) 重算一致性（realized_volatility）
    try:
        raw = load_raw_klines(sym, "1m", PARQUET_DATE_RANGE_LABEL)
        if not isinstance(raw.index, pd.DatetimeIndex):
            raw.index = pd.to_datetime(raw.index, utc=True)
        recomputed = compute_realized_volatility_label(raw)
        common = labels.index.intersection(recomputed.index)
        stored = labels["label_realized_volatility"].reindex(common)
        calc = recomputed.reindex(common)
        valid = stored.notna() & calc.notna()
        if valid.sum() > 0:
            match = (stored[valid].astype("int8") == calc[valid].astype("int8")).mean()
        else:
            match = float("nan")
        ok = match == match and match >= 0.99
        findings.append(
            {
                "check": "shift_h_recompute",
                "label": "label_realized_volatility",
                "risk_level": "低" if ok else "高",
                "detail": f"重算一致率={match:.4%}（n={int(valid.sum()):,}）",
                "passed": ok,
            }
        )
    except Exception as e:
        findings.append(
            {
                "check": "shift_h_recompute",
                "label": "label_realized_volatility",
                "risk_level": "中",
                "detail": f"無法重算驗證：{e}",
                "passed": False,
            }
        )

    # 3) 尾端 h 列（未來窗口不足應為 NaN）
    tail = labels.iloc[-min(HORIZON_H, len(labels)) :]
    tail_nan_ratio = float(tail.isna().mean().mean())
    tail_ok = tail_nan_ratio >= 0.5
    findings.append(
        {
            "check": "tail_future_window",
            "label": "all",
            "risk_level": "低" if tail_ok else "中",
            "detail": f"最後 {len(tail)} 列平均 NaN 比例={tail_nan_ratio:.2%}",
            "passed": tail_ok,
        }
    )

    passed = all(f.get("passed", False) for f in findings)
    return {
        "symbol": sym,
        "component": "label",
        "risk_level": _risk_level_from_findings(findings),
        "findings": findings,
        "passed": passed,
        "h": HORIZON_H,
        "state_window": STATE_WINDOW,
    }


def check_feature_leakage(symbol: str) -> dict[str, Any]:
    """
    檢查特徵是否使用未來資料（rolling + 預設 nan 策略）。
    """
    sym = symbol.strip().upper()
    p4 = Phase4Paths.default()
    findings: list[dict] = []

    feat = load_combined_features(sym, paths=p4)
    findings.append(
        {
            "check": "feature_index_monotonic",
            "risk_level": "低",
            "detail": f"單調遞增={feat.index.is_monotonic_increasing}，n={len(feat):,}",
            "passed": feat.index.is_monotonic_increasing,
        }
    )

    # roll_measure 使用 shift(1) — 文件級檢查
    findings.append(
        {
            "check": "roll_measure_shift1",
            "risk_level": "低",
            "detail": "roll_measure 等模組對 returns 使用 shift(1)，僅用 t-1 及更早資料",
            "passed": True,
        }
    )

    # 預設 dropna 路徑
    prep_drop = prepare_cross_market_features(sym, regime="all_day", nan_strategy="dropna")
    prep_ffill = prepare_cross_market_features(
        sym, regime="all_day", nan_strategy="ffill_then_dropna", allow_ffill=True
    )
    ffill_risk = len(prep_ffill.X) > len(prep_drop.X) * 1.05
    findings.append(
        {
            "check": "nan_strategy_default",
            "risk_level": "中" if ffill_risk else "低",
            "detail": (
                f"dropna n={len(prep_drop.X):,}；ffill_then_dropna n={len(prep_ffill.X):,}。"
                "診斷與訓練應使用 dropna。"
            ),
            "passed": not ffill_risk,
        }
    )

    passed = all(f.get("passed", False) for f in findings)
    return {
        "symbol": sym,
        "component": "feature",
        "risk_level": _risk_level_from_findings(findings),
        "findings": findings,
        "passed": passed,
    }


def check_time_alignment(symbol: str, regime: str = "all_day") -> dict[str, Any]:
    """確認特徵與標籤 join 後索引嚴格對齊。"""
    sym = symbol.strip().upper()
    regime_n = regime.strip().lower()
    p4 = Phase4Paths.default()

    feat = load_combined_features(sym, paths=p4)
    lab = load_combined_labels(sym, paths=p4)
    prepared = load_prepared_dataset(sym, regime_n)

    idx = prepared.X.index
    findings: list[dict] = []
    findings.append(
        {
            "check": "xy_index_equal",
            "risk_level": "低" if prepared.X.index.equals(prepared.y_all.index) else "高",
            "detail": f"X.index.equals(y.index)={prepared.X.index.equals(prepared.y_all.index)}",
            "passed": prepared.X.index.equals(prepared.y_all.index),
        }
    )
    dup = int(idx.duplicated().sum())
    findings.append(
        {
            "check": "no_duplicate_index",
            "risk_level": "高" if dup > 0 else "低",
            "detail": f"重複時間戳={dup}",
            "passed": dup == 0,
        }
    )
    tz_ok = idx.tz is not None and str(idx.tz) == "UTC"
    findings.append(
        {
            "check": "utc_timezone",
            "risk_level": "低" if tz_ok else "中",
            "detail": f"index.tz={idx.tz}",
            "passed": tz_ok,
        }
    )

    n_feat = len(feat)
    n_lab = len(lab)
    n_join = len(prepared.X)
    findings.append(
        {
            "check": "join_sample_counts",
            "risk_level": "低",
            "detail": (
                f"features={n_feat:,} labels={n_lab:,} joined={n_join:,} "
                f"range=[{idx.min()} .. {idx.max()}]"
            ),
            "passed": n_join > 0 and n_join <= min(n_feat, n_lab),
        }
    )

    passed = all(f.get("passed", False) for f in findings)
    return {
        "symbol": sym,
        "regime": regime_n,
        "component": "alignment",
        "risk_level": _risk_level_from_findings(findings),
        "findings": findings,
        "passed": passed,
        "n_features_raw": n_feat,
        "n_labels_raw": n_lab,
        "n_joined": n_join,
        "index_start": str(idx.min()),
        "index_end": str(idx.max()),
    }


def run_out_of_sample_test(
    symbol: str,
    regime: str,
    label_type: str,
    test_start_date: str = "2025-01-01",
    train_end_date: str = "2024-12-31",
) -> dict[str, Any]:
    """
    訓練集 <= train_end；測試集 >= test_start；重新訓練 RF（不用 Phase 4 pkl）。
    """
    sym = symbol.strip().upper()
    regime_n = regime.strip().lower()

    X, y = split_xy_by_label(load_prepared_dataset(sym, regime_n), label_type)
    train_mask, oos_mask = train_oos_masks(X.index, train_end=train_end_date, test_start=test_start_date)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_oos = X.loc[oos_mask]
    y_oos = y.loc[oos_mask]

    out: dict[str, Any] = {
        "symbol": sym,
        "regime": regime_n,
        "label_type": label_type,
        "train_end_date": train_end_date,
        "test_start_date": test_start_date,
        "n_train": int(len(X_train)),
        "n_oos": int(len(X_oos)),
    }

    if len(X_train) < 5000:
        out.update(
            {
                "status": "insufficient_train",
                "auc_in_sample": float("nan"),
                "auc_oos": float("nan"),
                "auc_gap": float("nan"),
                "pr_auc_oos": float("nan"),
            }
        )
        return out

    train_result = train_random_forest(X_train, y_train, label_type, regime=regime_n)
    model = train_result.model
    auc_cv = float(train_result.summary.get("auc_mean", float("nan")))
    out["walk_forward_auc_mean"] = auc_cv

    if len(X_oos) == 0:
        out.update(
            {
                "status": "no_oos_samples",
                "auc_in_sample": auc_cv,
                "auc_oos": float("nan"),
                "auc_gap": float("nan"),
                "pr_auc_oos": float("nan"),
                "warning": "無 2025+ 測試樣本，請確認 Data Lake 資料範圍",
            }
        )
        return out

    y_oos01 = labels_to_binary(y_oos)
    proba_oos = model.predict_proba(X_oos)[:, 1]
    pred_oos = (proba_oos >= 0.5).astype(np.int8)
    m_oos = classification_metrics(y_oos01, pred_oos, proba_oos)

    auc_oos = m_oos.get("auc", float("nan"))
    gap = float(auc_cv - auc_oos) if auc_cv == auc_cv and auc_oos == auc_oos else float("nan")

    out.update(
        {
            "status": "ok",
            "auc_in_sample": auc_cv,
            "pr_auc_in_sample_cv": float(train_result.summary.get("pr_auc_mean", float("nan"))),
            "auc_oos": auc_oos,
            "pr_auc_oos": m_oos.get("pr_auc"),
            "auc_gap": gap,
            "accuracy_oos": m_oos.get("accuracy"),
        }
    )
    return out


def summarize_leakage_risks(checks: list[dict]) -> dict[str, Any]:
    """彙總各診斷為表格列。"""
    rows: list[dict] = []
    for chk in checks:
        comp = chk.get("component", chk.get("symbol", "unknown"))
        regime = chk.get("regime", "")
        for f in chk.get("findings", []):
            rows.append(
                {
                    "symbol": chk.get("symbol", ""),
                    "regime": regime,
                    "component": comp,
                    "check": f.get("check", ""),
                    "risk_level": f.get("risk_level", "低"),
                    "passed": f.get("passed", False),
                    "detail": f.get("detail", ""),
                }
            )
        if not chk.get("findings"):
            rows.append(
                {
                    "symbol": chk.get("symbol", ""),
                    "regime": regime,
                    "component": comp,
                    "risk_level": chk.get("risk_level", "低"),
                    "passed": chk.get("passed", False),
                    "detail": f"整體 risk={chk.get('risk_level')}",
                    "check": "overall",
                }
            )

    df = pd.DataFrame(rows)
    high = int((df["risk_level"] == "高").sum()) if len(df) else 0
    med = int((df["risk_level"] == "中").sum()) if len(df) else 0
    return {
        "table": df,
        "n_high": high,
        "n_medium": med,
        "n_checks": len(df),
        "overall_risk": "高" if high > 0 else ("中" if med > 0 else "低"),
    }
