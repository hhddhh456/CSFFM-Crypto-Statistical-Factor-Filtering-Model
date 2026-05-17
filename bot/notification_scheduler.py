# -*- coding: utf-8 -*-
"""安全推播管線、fallback、APScheduler 四時段排程。"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from bot.config import NY_TZ, TAIWAN_TZ, UTC_TZ
from bot.model_predictor import get_full_prediction
from bot.notification_formatter import format_telegram_message
from bot.report_snapshot import save_report_snapshot
from bot.telegram_sender import send_telegram_message

logger = logging.getLogger(__name__)


class BotState:
    def __init__(self) -> None:
        self.last_push_utc: str | None = None
        self.last_trigger: str | None = None
        self.scheduler_running: bool = False


_API_FAIL_MSG = "即時行情 API 連線失敗，暫停方向性與期權建議。"


def build_fallback_report(error_message: str) -> dict:
    now_utc = datetime.now(ZoneInfo(UTC_TZ))
    now_tw = now_utc.astimezone(ZoneInfo(TAIWAN_TZ))
    now_ny = now_utc.astimezone(ZoneInfo(NY_TZ))

    feature_fail = "特徵與模型不一致" in error_message or "missing" in error_message.lower()
    api_fail = (
        _API_FAIL_MSG in error_message
        or "Binance" in error_message
        or "API" in error_message
        or "連線失敗" in error_message
    )
    if feature_fail:
        pause_msg = "即時特徵與訓練模型不一致，暫停方向性與期權建議。"
    elif api_fail:
        pause_msg = _API_FAIL_MSG
    else:
        pause_msg = "模型或資料異常，暫停方向性與期權建議。"

    return {
        "taiwan_time": now_tw.strftime("%Y-%m-%d %H:%M:%S"),
        "utc_time": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "ny_time": now_ny.strftime("%Y-%m-%d %H:%M:%S"),
        "btc_current_price": "N/A",
        "eth_current_price": "N/A",
        "btc_vol_dir": "暫停",
        "eth_vol_dir": "暫停",
        "btc_vol_prob": "N/A",
        "eth_vol_prob": "N/A",
        "btc_vol_magnitude": "N/A",
        "eth_vol_magnitude": "N/A",
        "btc_vol_range_str": "N/A",
        "eth_vol_range_str": "N/A",
        "btc_expected_min": "N/A",
        "btc_expected_max": "N/A",
        "eth_expected_min": "N/A",
        "eth_expected_max": "N/A",
        "btc_kurt_dir": "暫停",
        "eth_kurt_dir": "暫停",
        "btc_kurt_prob": "N/A",
        "eth_kurt_prob": "N/A",
        "btc_kurt_magnitude": "N/A",
        "eth_kurt_magnitude": "N/A",
        "btc_jb_dir": "暫停",
        "eth_jb_dir": "暫停",
        "btc_jb_prob": "N/A",
        "eth_jb_prob": "N/A",
        "btc_jb_magnitude": "N/A",
        "eth_jb_magnitude": "N/A",
        "btc_direction_advice": pause_msg,
        "eth_direction_advice": pause_msg,
        "btc_seq_dir": "暫停",
        "eth_seq_dir": "暫停",
        "btc_seq_prob": "N/A",
        "eth_seq_prob": "N/A",
        "btc_seller_bias": "N/A",
        "eth_seller_bias": "N/A",
        "btc_seller_bias_prob": "N/A",
        "eth_seller_bias_prob": "N/A",
        "btc_seller_bias_level": "N/A",
        "eth_seller_bias_level": "N/A",
        "btc_recommended_side": "暫停",
        "eth_recommended_side": "暫停",
        "btc_otm_level": "N/A",
        "eth_otm_level": "N/A",
        "btc_put_strike": "N/A",
        "btc_call_strike": "N/A",
        "eth_put_strike": "N/A",
        "eth_call_strike": "N/A",
        "btc_suggested_price_note": "資料異常，不提供履約價建議。",
        "eth_suggested_price_note": "資料異常，不提供履約價建議。",
        "btc_vol_strength": "N/A",
        "eth_vol_strength": "N/A",
        "btc_extreme_risk": "N/A",
        "eth_extreme_risk": "N/A",
        "btc_risk_note": "暫停交易建議。",
        "eth_risk_note": "暫停交易建議。",
        "btc_risk_structure_note": "請等待下一次有效模型輸出。",
        "eth_risk_structure_note": "請等待下一次有效模型輸出。",
        "data_freshness_note": pause_msg if api_fail else f"異常：{error_message}",
        "model_version": "Multi-timeframe RF",
    }


async def safe_generate_and_send_report(
    trigger_name: str = "manual",
    *,
    state: BotState | None = None,
) -> None:
    try:
        logger.info("Generating report, trigger=%s", trigger_name)
        data = get_full_prediction()
        data["trigger_name"] = trigger_name
        save_report_snapshot(data, prefix=f"{trigger_name}_normal")
        message = format_telegram_message(data)
        await send_telegram_message(message)
        if state is not None:
            from datetime import timezone

            state.last_push_utc = datetime.now(timezone.utc).isoformat()
            state.last_trigger = trigger_name
        logger.info("Report sent, trigger=%s", trigger_name)
    except Exception as e:
        err = str(e)
        if "延遲" in err or "stale" in err.lower():
            logger.warning(
                "Report fallback (stale data), trigger=%s: %s",
                trigger_name,
                err,
            )
        else:
            logger.exception("Report generation failed, trigger=%s", trigger_name)
        fallback_data = build_fallback_report(err)
        fallback_data["trigger_name"] = trigger_name
        save_report_snapshot(fallback_data, prefix=f"{trigger_name}_fallback")
        fallback_message = format_telegram_message(fallback_data)
        try:
            await send_telegram_message(fallback_message)
            if state is not None:
                from datetime import timezone

                state.last_push_utc = datetime.now(timezone.utc).isoformat()
                state.last_trigger = f"{trigger_name}_fallback"
        except Exception:
            logger.exception("Fallback report send failed")


async def send_test_report(*, state: BotState | None = None) -> None:
    await safe_generate_and_send_report(trigger_name="manual_test", state=state)


def setup_scheduler(state: BotState | None = None) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=UTC_TZ)

    async def _job(trigger_name: str) -> None:
        try:
            await safe_generate_and_send_report(trigger_name, state=state)
        except Exception:
            logger.exception("Scheduler job failed: %s", trigger_name)

    scheduler.add_job(
        _job,
        CronTrigger(hour=0, minute=30, timezone=UTC_TZ),
        kwargs={"trigger_name": "utc_0030"},
        id="utc_0030_report",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _job,
        CronTrigger(hour=7, minute=50, timezone=UTC_TZ),
        kwargs={"trigger_name": "utc_0750_settlement_report"},
        id="utc_0750_report",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _job,
        CronTrigger(hour=9, minute=30, timezone=NY_TZ),
        kwargs={"trigger_name": "ny_0930_open_report"},
        id="ny_0930_report",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _job,
        CronTrigger(hour=15, minute=30, timezone=NY_TZ),
        kwargs={"trigger_name": "ny_1530_preclose_report"},
        id="ny_1530_report",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    if state is not None:
        state.scheduler_running = True
    return scheduler
