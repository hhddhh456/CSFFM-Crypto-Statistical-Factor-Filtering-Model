# -*- coding: utf-8 -*-
"""報告 JSON 快照保存。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bot.config import REPORTS_DIR


def save_report_snapshot(data: dict, prefix: str = "report") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"{prefix}_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path
