# -*- coding: utf-8 -*-
"""Phase 5 回測視覺化（非互動 Agg 後端，避免 Windows Tk 崩潰）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(pnl_series: pd.Series, path: Path, *, title: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    equity = pnl_series.cumsum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(equity.index, equity.values, linewidth=1.0)
    ax.set_title(title or "Cumulative Strategy PnL (after fees)")
    ax.set_xlabel("open_time")
    ax.set_ylabel("cumulative_pnl")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    plt.close("all")
