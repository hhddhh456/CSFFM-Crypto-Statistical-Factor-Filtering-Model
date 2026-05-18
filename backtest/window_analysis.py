# -*- coding: utf-8 -*-
"""
Phase 2 多時間框架窗口（50–720 分鐘）成效比較：MDA 彙總 + 單窗口 OOS 消融。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.evaluation_utils import (
    LABEL_ORDER,
    BacktestPaths,
    classification_metrics,
    labels_to_binary,
    load_prepared_dataset,
    normalize_regime,
    split_xy_by_label,
    train_oos_masks,
)
from features.feature_utils import WINDOWS
from models.random_forest_trainer import train_random_forest
from models.regime_utils import ALL_REGIMES, regime_folder_name

_WINDOW_SUFFIX_RE = re.compile(r"_(\d+)$")
_EXPECTED_COLS_PER_WINDOW = 10  # btc_/eth_ × 5 指標
_EXPECTED_OOS_ROWS = 200  # 2 symbols × 4 regimes × 5 labels × 5 windows


def extract_window_from_feature(feature: str) -> int | None:
    m = _WINDOW_SUFFIX_RE.search(str(feature).strip())
    if not m:
        return None
    w = int(m.group(1))
    return w if w in WINDOWS else None


def select_columns_for_window(X: pd.DataFrame, window: int) -> pd.DataFrame:
    if window not in WINDOWS:
        raise ValueError(f"非法窗口 {window}，可用：{WINDOWS}")
    suffix = f"_{window}"
    cols = sorted(c for c in X.columns if c.endswith(suffix))
    if len(cols) != _EXPECTED_COLS_PER_WINDOW:
        raise ValueError(
            f"窗口 {window} 預期 {_EXPECTED_COLS_PER_WINDOW} 欄，實際 {len(cols)}：{cols}"
        )
    return X[cols]


def _parse_mda_path(path: Path) -> tuple[str, str, str] | None:
    """backtest/BTCUSDT/all_day/mda_label_foo.csv -> symbol, regime, label_type."""
    parts = path.parts
    if len(parts) < 4 or not path.name.startswith("mda_label_"):
        return None
    symbol = parts[-3].upper()
    folder = parts[-2]
    label_type = path.stem.replace("mda_", "", 1)
    folder_to_regime = {regime_folder_name(r): normalize_regime(r) for r in ALL_REGIMES}
    regime = folder_to_regime.get(folder)
    if regime is None:
        return None
    return symbol, regime, label_type


def aggregate_mda_windows(paths: BacktestPaths | None = None) -> pd.DataFrame:
    """
    掃描所有 mda_label_*.csv，按 window 聚合 importance_mean。
    回傳細粒度表（含 overall 彙總列 granularity=overall）。
    """
    p = paths or BacktestPaths.default()
    rows: list[dict] = []

    for mda_path in sorted(p.root.glob("*/*/mda_label_*.csv")):
        parsed = _parse_mda_path(mda_path)
        if parsed is None:
            continue
        symbol, regime, label_type = parsed
        try:
            mda = pd.read_csv(mda_path, encoding="utf-8-sig")
        except Exception:
            continue
        if "feature" not in mda.columns or "importance_mean" not in mda.columns:
            continue
        for _, r in mda.iterrows():
            w = extract_window_from_feature(r["feature"])
            if w is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "regime": regime,
                    "label_type": label_type,
                    "window": w,
                    "importance_mean": float(r["importance_mean"]),
                    "importance_std": float(r.get("importance_std", float("nan"))),
                    "feature": r["feature"],
                    "granularity": "feature",
                }
            )

    if not rows:
        return pd.DataFrame()

    feat_df = pd.DataFrame(rows)
    agg_rows: list[dict] = []

    def _add_agg(group_cols: list[str], granularity: str) -> None:
        g = (
            feat_df.groupby(group_cols, as_index=False)["importance_mean"]
            .agg(importance_mean_sum="sum", importance_mean_avg="mean", n_features="count")
        )
        rank_cols = [c for c in group_cols if c != "window"]
        if rank_cols:
            g = g.sort_values(rank_cols + ["importance_mean_avg"], ascending=[True] * len(rank_cols) + [False])
            g["window_rank_mda"] = g.groupby(rank_cols, group_keys=False).cumcount() + 1
        else:
            g = g.sort_values("importance_mean_avg", ascending=False).reset_index(drop=True)
            g["window_rank_mda"] = range(1, len(g) + 1)
        for _, row in g.iterrows():
            rec = {c: row[c] for c in group_cols}
            rec.update(
                {
                    "importance_mean_sum": row["importance_mean_sum"],
                    "importance_mean_avg": row["importance_mean_avg"],
                    "n_features": int(row["n_features"]),
                    "window_rank_mda": int(row["window_rank_mda"]),
                    "granularity": granularity,
                }
            )
            agg_rows.append(rec)

    for sym in feat_df["symbol"].unique():
        for reg in feat_df["regime"].unique():
            for lbl in feat_df["label_type"].unique():
                _add_agg(["symbol", "regime", "label_type", "window"], "by_combo")
    _add_agg(["symbol", "regime", "window"], "by_symbol_regime")
    _add_agg(["regime", "window"], "by_regime")
    _add_agg(["window"], "overall")

    detail = feat_df.copy()
    out = pd.concat([detail, pd.DataFrame(agg_rows)], ignore_index=True, sort=False)
    return out


def evaluate_window_ablation_oos(
    symbol: str,
    label_type: str,
    regime: str,
    window: int,
    *,
    test_start: str = "2025-01-01",
    train_end: str = "2024-12-31",
) -> dict:
    """單一窗口 10 維特徵的嚴格 OOS 評估。"""
    sym = symbol.strip().upper()
    regime_n = normalize_regime(regime)

    X, y = split_xy_by_label(load_prepared_dataset(sym, regime_n), label_type)
    X = select_columns_for_window(X, window)
    train_mask, oos_mask = train_oos_masks(X.index, train_end=train_end, test_start=test_start)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_oos = X.loc[oos_mask]
    y_oos = y.loc[oos_mask]

    report: dict = {
        "symbol": sym,
        "label_type": label_type,
        "regime": regime_n,
        "window": int(window),
        "train_end": train_end,
        "test_start": test_start,
        "n_train": int(len(X_train)),
        "n_oos": int(len(X_oos)),
        "auc_in_sample": float("nan"),
        "auc_oos": float("nan"),
        "pr_auc_oos": float("nan"),
        "auc_gap": float("nan"),
        "status": "pending",
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
    }

    if len(X_train) < 5000:
        report["status"] = "insufficient_train"
        return report

    train_result = train_random_forest(X_train, y_train, label_type, regime=regime_n)
    auc_cv = float(train_result.summary.get("auc_mean", float("nan")))
    report["auc_in_sample"] = auc_cv

    if len(X_oos) == 0:
        report["status"] = "no_oos_samples"
        return report

    y_oos01 = labels_to_binary(y_oos)
    proba_oos = train_result.model.predict_proba(X_oos)[:, 1]
    pred_oos = (proba_oos >= 0.5).astype(np.int8)
    m_oos = classification_metrics(y_oos01, pred_oos, proba_oos)
    auc_oos = m_oos.get("auc", float("nan"))
    report["auc_oos"] = auc_oos
    report["pr_auc_oos"] = m_oos.get("pr_auc", float("nan"))
    if auc_cv == auc_cv and auc_oos == auc_oos:
        report["auc_gap"] = float(auc_cv - auc_oos)
    report["status"] = "ok"
    return report


def run_window_ablation_oos(
    *,
    test_start: str = "2025-01-01",
    train_end: str = "2024-12-31",
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    regimes: tuple[str, ...] | None = None,
    label_types: tuple[str, ...] | None = None,
    force_recompute: bool = False,
    paths: BacktestPaths | None = None,
) -> pd.DataFrame:
    p = paths or BacktestPaths.default()
    out_csv = p.window_oos_comparison_path()
    regime_list = regimes or ALL_REGIMES
    labels = label_types or tuple(LABEL_ORDER)

    if out_csv.is_file() and not force_recompute:
        existing = pd.read_csv(out_csv, encoding="utf-8-sig")
        if len(existing) >= _EXPECTED_OOS_ROWS:
            ok = existing[existing.get("status", "") == "ok"] if "status" in existing.columns else existing
            if len(ok) >= _EXPECTED_OOS_ROWS * 0.9:
                print(f"[SKIP] 窗口 OOS 已存在：{out_csv}（{len(existing)} 列）")
                return existing

    records: list[dict] = []
    total = len(symbols) * len(regime_list) * len(labels) * len(WINDOWS)
    done = 0

    for symbol in symbols:
        for regime in regime_list:
            regime_n = normalize_regime(regime)
            for label_type in labels:
                for window in WINDOWS:
                    done += 1
                    print(
                        f"  [{done}/{total}] {symbol} {regime_n} {label_type} window={window}..."
                    )
                    try:
                        rec = evaluate_window_ablation_oos(
                            symbol,
                            label_type,
                            regime_n,
                            window,
                            test_start=test_start,
                            train_end=train_end,
                        )
                        records.append(rec)
                        if rec.get("status") == "ok":
                            print(f"    AUC OOS={rec.get('auc_oos', float('nan')):.4f}")
                    except Exception as e:
                        records.append(
                            {
                                "symbol": symbol,
                                "label_type": label_type,
                                "regime": regime_n,
                                "window": window,
                                "status": "error",
                                "error": str(e),
                            }
                        )
                        print(f"    [WARN] {e}")

    df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] {out_csv}")
    return df


def _rank_windows_by_oos(oos_df: pd.DataFrame) -> pd.DataFrame:
    ok = oos_df[oos_df["status"] == "ok"].copy() if "status" in oos_df.columns else oos_df
    if ok.empty or "auc_oos" not in ok.columns:
        return pd.DataFrame()
    ok["window"] = ok["window"].astype(int)
    summary = (
        ok.groupby("window", as_index=False)
        .agg(
            mean_auc_oos=("auc_oos", "mean"),
            median_auc_oos=("auc_oos", "median"),
            mean_pr_auc_oos=("pr_auc_oos", "mean"),
            n_ok=("auc_oos", "count"),
        )
        .sort_values("mean_auc_oos", ascending=False)
    )
    summary["oos_rank"] = range(1, len(summary) + 1)
    return summary


def _rank_windows_by_mda(mda_df: pd.DataFrame) -> pd.DataFrame:
    overall = mda_df[mda_df["granularity"] == "overall"].copy() if "granularity" in mda_df.columns else pd.DataFrame()
    if overall.empty:
        by = mda_df[mda_df.get("granularity", "") == "by_combo"] if "granularity" in mda_df.columns else mda_df
        if by.empty:
            return pd.DataFrame()
        overall = (
            by.groupby("window", as_index=False)["importance_mean_avg"]
            .mean()
            .rename(columns={"importance_mean_avg": "importance_mean_avg"})
        )
    else:
        overall = overall[["window", "importance_mean_avg", "importance_mean_sum", "window_rank_mda"]].copy()
    overall = overall.sort_values(
        "importance_mean_avg" if "importance_mean_avg" in overall.columns else "importance_mean_sum",
        ascending=False,
    )
    overall["mda_rank"] = range(1, len(overall) + 1)
    return overall


def write_window_analysis_summary(
    mda_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    paths: BacktestPaths | None = None,
    *,
    test_start: str = "2025-01-01",
) -> Path:
    p = paths or BacktestPaths.default()
    out = p.window_analysis_summary_path()

    oos_rank = _rank_windows_by_oos(oos_df)
    mda_rank = _rank_windows_by_mda(mda_df)

    lines = [
        "# 多時間框架窗口比較（50 / 100 / 240 / 480 / 720 分鐘）",
        "",
        f"> 嚴格 OOS：train <= 2024-12-31，test >= {test_start} UTC。",
        "> **主 KPI**：單窗口消融 OOS AUC（每次僅用該窗口 10 維特徵重訓 RF）。",
        "> **輔助**：全 50 維模型 MDA 按窗口彙總（特徵重要性，非單獨預測力）。",
        "",
        f"- 細項 CSV：`window_oos_comparison.csv`（{len(oos_df)} 列）",
        f"- MDA 彙總：`window_mda_summary.csv`",
        "",
    ]

    if not oos_rank.empty:
        best = int(oos_rank.iloc[0]["window"])
        worst = int(oos_rank.iloc[-1]["window"])
        lines.append("## OOS 消融排名（主 KPI）")
        lines.append("")
        lines.append("| 排名 | 窗口(分) | 平均 AUC | 中位 AUC | 平均 PR-AUC | n |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in oos_rank.iterrows():
            lines.append(
                f"| {int(row['oos_rank'])} | {int(row['window'])} | "
                f"{row['mean_auc_oos']:.4f} | {row['median_auc_oos']:.4f} | "
                f"{row['mean_pr_auc_oos']:.4f} | {int(row['n_ok'])} |"
            )
        lines.append("")
        lines.append(
            f"**結論（OOS）**：整體最佳窗口為 **{best} 分鐘**（平均 AUC {oos_rank.iloc[0]['mean_auc_oos']:.4f}），"
            f"最弱為 **{worst} 分鐘**（{oos_rank.iloc[-1]['mean_auc_oos']:.4f}）。"
        )
        lines.append("")

        ok = oos_df[oos_df["status"] == "ok"] if "status" in oos_df.columns else oos_df
        if not ok.empty and "regime" in ok.columns:
            lines.append("### 各 Regime 最佳窗口（平均 OOS AUC）")
            lines.append("")
            for reg in sorted(ok["regime"].unique()):
                sub = ok[ok["regime"] == reg]
                by_w = sub.groupby("window")["auc_oos"].mean().sort_values(ascending=False)
                if by_w.empty:
                    continue
                lines.append(f"- **{reg}**：{int(by_w.index[0])} 分（AUC {by_w.iloc[0]:.4f}）")
            lines.append("")

        baseline_path = p.oos_comparison_path()
        if baseline_path.is_file():
            bl = pd.read_csv(baseline_path, encoding="utf-8-sig")
            if "auc_oos" in bl.columns and bl["status"].eq("ok").any() if "status" in bl.columns else True:
                bl_ok = bl[bl["status"] == "ok"] if "status" in bl.columns else bl
                full_mean = float(bl_ok["auc_oos"].mean()) if len(bl_ok) else float("nan")
                win_mean = float(oos_rank.iloc[0]["mean_auc_oos"])
                if full_mean == full_mean:
                    lines.append(
                        f"### 對照全 50 維 OOS（`oos_comparison.csv`）\n\n"
                        f"- 全特徵平均 OOS AUC：**{full_mean:.4f}**\n"
                        f"- 最佳單窗口（{best} 分）平均 OOS AUC：**{win_mean:.4f}** "
                        f"（{'高於' if win_mean > full_mean else '低於'}全特徵）\n"
                    )
                    lines.append("")
    else:
        lines.append("_尚無有效 OOS 窗口消融結果。_\n")

    if not mda_rank.empty:
        lines.append("## MDA 按窗口彙總（輔助）")
        lines.append("")
        col = "importance_mean_avg" if "importance_mean_avg" in mda_rank.columns else "importance_mean_sum"
        lines.append("| MDA 排名 | 窗口(分) | 平均 importance |")
        lines.append("| --- | --- | --- |")
        for _, row in mda_rank.iterrows():
            lines.append(
                f"| {int(row['mda_rank'])} | {int(row['window'])} | {row[col]:.6f} |"
            )
        lines.append("")
        mda_best = int(mda_rank.iloc[0]["window"])
        if not oos_rank.empty:
            oos_best = int(oos_rank.iloc[0]["window"])
            if mda_best == oos_best:
                lines.append(f"MDA 與 OOS 排名一致：皆以 **{mda_best} 分鐘** 居前。")
            else:
                lines.append(
                    f"MDA 最高窗口為 **{mda_best} 分**，OOS 消融最高為 **{oos_best} 分**；"
                    "可能因特徵共線或全模型依賴多窗口互補。"
                )
        lines.append("")

    lines.append("## 解讀指引")
    lines.append("")
    lines.append("1. 部署模型使用全部 50 維；單窗口 OOS 回答「若只用該尺度，2025+ 預測力如何」。")
    lines.append("2. MDA 回答「全模型中哪個尺度的欄位最常被依賴」。")
    lines.append("3. UTC regime（asia/settlement 等）與特徵窗口為正交維度，可組合使用。")
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {out}")
    return out


def run_window_analysis(
    *,
    test_start: str = "2025-01-01",
    train_end: str = "2024-12-31",
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    regimes: tuple[str, ...] | None = None,
    label_types: tuple[str, ...] | None = None,
    force_recompute: bool = False,
    paths: BacktestPaths | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """完整窗口分析：MDA 彙總 → OOS 消融 → summary markdown。"""
    p = paths or BacktestPaths.default()

    print("\n========== 窗口分析：MDA 彙總 ==========")
    mda_df = aggregate_mda_windows(p)
    mda_csv = p.window_mda_summary_path()
    mda_csv.parent.mkdir(parents=True, exist_ok=True)
    mda_df.to_csv(mda_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] {mda_csv}（{len(mda_df)} 列）")

    print("\n========== 窗口分析：OOS 消融 ==========")
    oos_df = run_window_ablation_oos(
        test_start=test_start,
        train_end=train_end,
        symbols=symbols,
        regimes=regimes,
        label_types=label_types,
        force_recompute=force_recompute,
        paths=p,
    )

    write_window_analysis_summary(mda_df, oos_df, p, test_start=test_start)
    return mda_df, oos_df
