# -*- coding: utf-8 -*-
"""Phase 6 集中設定（Token 僅來自專案根目錄 .env）。"""

from __future__ import annotations

import os
from pathlib import Path

from utils.config import load_project_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OOS_COMPARISON_CSV = PROJECT_ROOT / "backtest" / "oos_comparison.csv"
BACKTEST_ROOT = PROJECT_ROOT / "backtest"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

TAIWAN_TZ = "Asia/Taipei"
NY_TZ = "America/New_York"
UTC_TZ = "UTC"

MAX_TELEGRAM_MESSAGE_LENGTH = 3900

# V3 — Real-Time Runtime Inference
RUNTIME_VERSION = "V3"
RAW_KLINE_LIMIT = 1600
FEATURE_WINDOW = 1500
RUNTIME_FEATURE_SOURCE = "binance_1600_klines"

PREDICTION_REGIME = "all_day"
HORIZON_MINUTES = 1500
PROBA_HIGH_CONF = 0.55
OOS_AUC_MIN = 0.55
DIRECTIONAL_THRESHOLD = 0.62
HIGH_CONVICTION_THRESHOLD = 0.68
FRESHNESS_MAX_SECONDS = 180
DAILY_VOL_MAX = 0.25
TAIL_RISK_HIGH = 0.7

LABELS_FOR_REPORT = (
    "label_realized_volatility",
    "label_kurtosis",
    "label_jarque_bera",
    "label_sequential_correlation",
)


def _ensure_env_loaded() -> None:
    load_project_env()


def get_telegram_token() -> str:
    _ensure_env_loaded()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token or token.startswith("your_"):
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")
    return token


def get_telegram_chat_id() -> str:
    file_vals = load_project_env()
    cid = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not cid:
        cid = (file_vals.get("TELEGRAM_CHAT_ID") or "").strip()
    if not cid:
        raise RuntimeError("Missing TELEGRAM_CHAT_ID in .env")
    return cid


def get_log_level() -> str:
    _ensure_env_loaded()
    return os.environ.get("LOG_LEVEL", "INFO").strip().upper()


def allow_stale_market_data() -> bool:
    """開發用：BOT_ALLOW_STALE_DATA=1 時略過 K 線 180s 檢查（仍使用 Data Lake 末筆）。"""
    _ensure_env_loaded()
    return os.environ.get("BOT_ALLOW_STALE_DATA", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def strike_tick(symbol: str) -> int:
    _ensure_env_loaded()
    base = symbol.strip().upper().replace("USDT", "")
    if base == "BTC":
        return int(os.environ.get("STRIKE_TICK_BTC", "500"))
    if base == "ETH":
        return int(os.environ.get("STRIKE_TICK_ETH", "50"))
    return 10


# 模組匯入時驗證（延遲：僅在首次取 token 時）
TELEGRAM_BOT_TOKEN: str | None = None
TELEGRAM_CHAT_ID: str | None = None


def get_bot_credentials() -> tuple[str, str]:
    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    if TELEGRAM_BOT_TOKEN is None:
        TELEGRAM_BOT_TOKEN = get_telegram_token()
    if TELEGRAM_CHAT_ID is None:
        TELEGRAM_CHAT_ID = get_telegram_chat_id()
    return TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
