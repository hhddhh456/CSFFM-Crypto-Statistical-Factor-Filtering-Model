# -*- coding: utf-8 -*-
"""aiogram 3.x 指令（§19）。"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, TelegramObject

from bot.config import get_telegram_token
from bot.live_feature_pipeline import build_live_snapshot
from bot.model_predictor import get_full_prediction, get_model_status_audit
from bot.notification_formatter import format_predict_reply, format_status, safe_text
from bot.notification_scheduler import BotState, safe_generate_and_send_report

logger = logging.getLogger(__name__)
router = Router(name="csffm_commands")


class BotStateMiddleware(BaseMiddleware):
    def __init__(self, state: BotState) -> None:
        self.state = state

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        data["bot_state"] = self.state
        return await handler(event, data)


def create_bot() -> Bot:
    return Bot(
        token=get_telegram_token(),
        default=DefaultBotProperties(parse_mode="HTML"),
    )


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    logger.info("Command /start from chat_id=%s", message.chat.id)
    await message.answer(
        "CSFFM Telegram Bot 已啟動。\n輸入 /help 查看可用指令。"
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    logger.info("Command /help from chat_id=%s", message.chat.id)
    await message.answer(
        "/status - 查看 Bot 狀態\n"
        "/predict_btc - 即時 BTC 預測\n"
        "/predict_eth - 即時 ETH 預測\n"
        "/test_report - 立即發送完整測試報告\n"
        "/help - 說明"
    )


@router.message(Command("status"))
async def cmd_status(message: Message, bot_state: BotState) -> None:
    logger.info("Command /status from chat_id=%s", message.chat.id)
    snap = None
    try:
        snap = build_live_snapshot()
    except Exception as e:
        logger.warning("status snapshot failed: %s", e)
    model_audit = get_model_status_audit(snap)
    if snap is not None:
        from bot.feature_drift import feature_hash

        model_audit = {**model_audit, "feature_hash": feature_hash(snap.X_row)[:8]}
    text = format_status(
        last_push=bot_state.last_push_utc,
        last_trigger=bot_state.last_trigger,
        scheduler_running=bot_state.scheduler_running,
        snap=snap,
        model_audit=model_audit,
    )
    await message.answer(text, parse_mode="HTML")


@router.message(Command("predict_btc"))
async def cmd_predict_btc(message: Message, bot_state: BotState) -> None:
    _ = bot_state
    logger.info("Command /predict_btc from chat_id=%s", message.chat.id)
    try:
        data = get_full_prediction()
        await message.answer(format_predict_reply(data, "btc"))
    except Exception as e:
        logger.warning("Predict BTC failed: %s", e)
        await message.answer(f"模型暫時無法輸出預測：{safe_text(str(e))}")


@router.message(Command("predict_eth"))
async def cmd_predict_eth(message: Message, bot_state: BotState) -> None:
    _ = bot_state
    logger.info("Command /predict_eth from chat_id=%s", message.chat.id)
    try:
        data = get_full_prediction()
        await message.answer(format_predict_reply(data, "eth"))
    except Exception as e:
        logger.warning("Predict ETH failed: %s", e)
        await message.answer(f"模型暫時無法輸出預測：{safe_text(str(e))}")


@router.message(Command("test_report", "testpush"))
async def cmd_test_report(message: Message, bot_state: BotState) -> None:
    logger.info("Command /test_report from chat_id=%s", message.chat.id)
    await message.answer("開始產生測試報告。")
    await safe_generate_and_send_report(
        trigger_name="telegram_manual_test",
        state=bot_state,
    )
    await message.answer("測試報告流程已執行。")


@router.message()
async def on_any_message(message: Message) -> None:
    """非指令文字：避免 aiogram 顯示 not handled。"""
    if not message.text:
        return
    text = message.text.strip()
    if text.startswith("/"):
        await message.answer(
            "未知指令。輸入 /help 查看可用指令。"
        )
        return
    await message.answer("請使用 /help 查看可用指令（例如 /status、/predict_btc）。")


def create_dispatcher(state: BotState) -> Dispatcher:
    dp = Dispatcher()
    dp.update.middleware(BotStateMiddleware(state))
    dp.include_router(router)
    return dp
