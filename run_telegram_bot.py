# -*- coding: utf-8 -*-
"""
Phase 6 主控：CSFFM Telegram Bot + 排程推播。

用法：
  python run_telegram_bot.py
  python run_telegram_bot.py --test
  python run_telegram_bot.py --test-push
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from bot.config import get_telegram_chat_id
from bot.logging_config import setup_logging
from bot.notification_scheduler import BotState, send_test_report, setup_scheduler
from bot.telegram_bot import create_bot, create_dispatcher

logger = logging.getLogger("run_telegram_bot")


async def main(*, test_mode: bool = False) -> None:
    setup_logging()

    if test_mode:
        logger.info("Running test report mode")
        from bot.model_predictor import get_full_prediction

        try:
            data = get_full_prediction()
            logger.info("Feature consistency: %s", data.get("_feature_consistency"))
            logger.info("Model audit: %s", data.get("_model_audit"))
        except Exception:
            logger.exception("get_full_prediction failed in --test")
        await send_test_report()
        return

    bot = create_bot()
    state = BotState()
    dp = create_dispatcher(state)

    scheduler = None
    if get_telegram_chat_id():
        scheduler = setup_scheduler(state)
        scheduler.start()
        state.scheduler_running = True
        logger.info(
            "排程已啟動：00:30 UTC、07:50 UTC、09:30/15:30 America/New_York"
        )
    else:
        logger.warning("未設定 TELEGRAM_CHAT_ID，僅啟動指令模式（無排程推播）。")

    logger.info("Bot polling 啟動中（模型=all_day）...")
    try:
        await dp.start_polling(
            bot,
            drop_pending_updates=True,
            allowed_updates=dp.resolve_used_update_types(),
        )
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)
        await bot.session.close()
        logger.info("Bot shutdown completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSFFM Telegram Bot")
    parser.add_argument("--test", action="store_true", help="立即測試推播後退出")
    parser.add_argument("--test-push", action="store_true", help="同 --test")
    args = parser.parse_args()
    try:
        asyncio.run(main(test_mode=args.test or args.test_push))
    except KeyboardInterrupt:
        sys.exit(0)
