# -*- coding: utf-8 -*-
"""Phase 6 logging 設定。"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from bot.config import LOGS_DIR, get_log_level


def setup_logging(level: str | None = None) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lvl = level or get_log_level()
    root = logging.getLogger()
    root.setLevel(getattr(logging, lvl, logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOGS_DIR / "telegram_bot.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)
