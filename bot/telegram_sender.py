# -*- coding: utf-8 -*-
"""Telegram 發送（HTML + tenacity 重試）。"""

from __future__ import annotations

import logging

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from tenacity import retry, stop_after_attempt, wait_exponential

from bot.config import get_bot_credentials

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def send_telegram_message(message: str) -> None:
    token, chat_id = get_bot_credentials()
    bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            disable_web_page_preview=True,
        )
        logger.info("Telegram message sent successfully")
    finally:
        await bot.session.close()
