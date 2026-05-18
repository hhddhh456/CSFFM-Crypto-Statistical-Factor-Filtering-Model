# -*- coding: utf-8 -*-
"""Telegram HTML 報告格式化（§13）。"""

from __future__ import annotations

import html
from typing import Any

from bot.config import MAX_TELEGRAM_MESSAGE_LENGTH, PREDICTION_REGIME
from bot.live_feature_pipeline import LiveSnapshot


def safe_text(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def safe_num(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_magnitude(value: object) -> str:
    """指標行尾端幅度；缺值時為「幅度 N/A」。"""
    if value is None or value == "":
        return "幅度 N/A"
    return f"幅度 {safe_text(value)}"


def compact_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def truncate_message(message: str, max_len: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> str:
    if len(message) <= max_len:
        return message
    suffix = "\n\n⚠️ 訊息過長，部分內容已截斷。"
    return message[: max_len - len(suffix)] + suffix


def _finalize_message(message: str) -> str:
    return truncate_message(compact_lines(message))


def _direction_advice_lines(data: dict[str, Any], prefix: str) -> list[str]:
    pfx = prefix
    return [
        f"• 極端風險：峰度（Kurtosis）：{safe_text(data.get(f'{pfx}_kurt_dir', 'N/A'))}"
        f"｜Jarque-Bera 檢定：{safe_text(data.get(f'{pfx}_jb_dir', 'N/A'))}",
        f"• 序列相關（Sequential Correlation）：{safe_text(data.get(f'{pfx}_seq_dir', 'N/A'))}"
        f"（信心 {safe_text(data.get(f'{pfx}_seq_prob', 'N/A'))}%）",
        f"• 賣方輔助：{safe_text(data.get(f'{pfx}_seller_bias', 'N/A'))}"
        f"（{safe_text(data.get(f'{pfx}_seller_bias_prob', 'N/A'))}%，"
        f"{safe_text(data.get(f'{pfx}_seller_bias_level', 'N/A'))}）",
    ]


def _format_asset_section(data: dict[str, Any], prefix: str, title: str) -> str:
    pfx = prefix
    lines = [
        f"📍 <b>【{title}】</b>",
        f"現價：${safe_num(data.get(f'{pfx}_current_price'), 2)}",
        "📊 <b>指標預測</b>",
        (
            f"• Realized Volatility：{safe_text(data.get(f'{pfx}_vol_dir'))}"
            f"｜信心 {safe_text(data.get(f'{pfx}_vol_prob'))}%"
            f"｜{format_magnitude(data.get(f'{pfx}_vol_magnitude'))}"
        ),
        f"• 常態預期 24H 波動：<b>{safe_text(data.get(f'{pfx}_vol_range_str'))}</b>",
        (
            f"• 預期 24H 價格區間：${safe_num(data.get(f'{pfx}_expected_min'), 2)}"
            f" ~ ${safe_num(data.get(f'{pfx}_expected_max'), 2)}"
        ),
        (
            f"• Kurtosis 尾部風險：{safe_text(data.get(f'{pfx}_kurt_dir'))}"
            f"｜信心 {safe_text(data.get(f'{pfx}_kurt_prob'))}%"
            f"｜{format_magnitude(data.get(f'{pfx}_kurt_magnitude'))}"
        ),
        (
            f"• Jarque-Bera 分佈異常：{safe_text(data.get(f'{pfx}_jb_dir'))}"
            f"｜信心 {safe_text(data.get(f'{pfx}_jb_prob'))}%"
            f"｜{format_magnitude(data.get(f'{pfx}_jb_magnitude'))}"
        ),
        "━━━━━━━━━━━━━━",
        *_direction_advice_lines(data, pfx),
        "━━━━━━━━━━━━━━",
        "⚠️ <b>期權交易建議</b>",
        f"• 建議賣方策略：{safe_text(data.get(f'{pfx}_recommended_side'))}",
        f"• OTM 等級：{safe_text(data.get(f'{pfx}_otm_level'))}",
        (
            f"• 建議賣出價格：PUT {safe_text(data.get(f'{pfx}_put_strike'))}"
            f"｜CALL {safe_text(data.get(f'{pfx}_call_strike'))}"
        ),
        f"• 提示：{safe_text(data.get(f'{pfx}_suggested_price_note'))}",
        (
            f"• 理由：波動幅度 {safe_text(data.get(f'{pfx}_vol_strength'))}"
            f" + 極端風險 {safe_text(data.get(f'{pfx}_extreme_risk'))}"
        ),
        f"• 風險提示：{safe_text(data.get(f'{pfx}_risk_note'))}",
        f"• 結構建議：{safe_text(data.get(f'{pfx}_risk_structure_note'))}",
    ]
    return "\n".join(lines)


def format_telegram_message(data: dict[str, Any]) -> str:
    """傳入完整預測 dict；動態欄位皆 escape。Telegram parse_mode=HTML。"""
    model_version = data.get("model_version", "Multi-timeframe RF")
    if isinstance(model_version, str) and "（" in model_version:
        model_version = model_version.split("（", 1)[0].strip()
    sections = [
        "🚨 <b>CSFFM 預測報告</b>",
        f"🕒 臺灣時間：{safe_text(data.get('taiwan_time'))}",
        f"UTC：{safe_text(data.get('utc_time'))}",
        f"美東：{safe_text(data.get('ny_time'))}",
        "━━━━━━━━━━━━━━",
        _format_asset_section(data, "btc", "BTC"),
        "━━━━━━━━━━━━━━",
        _format_asset_section(data, "eth", "ETH"),
        "━━━━━━━━━━━━━━",
        f"資料更新：{safe_text(data.get('data_freshness_note', '最新 1 分鐘 K 線'))}",
        f"模型版本：{safe_text(model_version)}",
    ]
    vpin_line = data.get("vpin_report_line")
    if vpin_line:
        sections.append(safe_text(vpin_line))
    sections.append("免責聲明：本報告由統計模型產生，僅供參考，不構成投資建議。")
    return _finalize_message("\n".join(sections))


def format_predict_reply(data: dict[str, Any], symbol: str) -> str:
    """/predict_btc、/predict_eth 精簡回覆。"""
    pfx = symbol.lower()
    sym_label = symbol.upper()
    mag = data.get(f"{pfx}_vol_magnitude", "N/A")
    mag_display = safe_text(mag) if mag and str(mag).startswith(("+", "-")) else safe_text(mag)
    lines = [
        f"📍 {sym_label} 即時預測",
        f"現價：${safe_num(data.get(f'{pfx}_current_price'), 2)}",
        (
            f"波動（Volatility）：{safe_text(data.get(f'{pfx}_vol_dir'))}"
            f"｜{safe_text(data.get(f'{pfx}_vol_prob'))}%"
            f"｜幅度 {mag_display}"
        ),
        (
            f"24H 區間（Range）：${safe_num(data.get(f'{pfx}_expected_min'), 2)}"
            f" ~ ${safe_num(data.get(f'{pfx}_expected_max'), 2)}"
        ),
        (
            f"• 極端風險：峰度（Kurtosis）：{safe_text(data.get(f'{pfx}_kurt_dir'))}"
            f"｜Jarque-Bera 檢定：{safe_text(data.get(f'{pfx}_jb_dir'))}"
        ),
        (
            f"• 序列相關（Sequential Correlation）：{safe_text(data.get(f'{pfx}_seq_dir'))}"
            f"（信心 {safe_text(data.get(f'{pfx}_seq_prob'))}%）"
        ),
        (
            f"• 賣方輔助：{safe_text(data.get(f'{pfx}_seller_bias'))}"
            f"（{safe_text(data.get(f'{pfx}_seller_bias_prob'))}%，"
            f"{safe_text(data.get(f'{pfx}_seller_bias_level'))}）"
        ),
        (
            f"建議履約價（Strike）：PUT {safe_text(data.get(f'{pfx}_put_strike'))}"
            f"｜CALL {safe_text(data.get(f'{pfx}_call_strike'))}"
        ),
        "免責聲明：本報告由統計模型產生，僅供參考，不構成投資建議。",
    ]
    return _finalize_message("\n".join(lines))


def format_status(
    *,
    last_push: str | None,
    last_trigger: str | None,
    scheduler_running: bool,
    snap: LiveSnapshot | None,
    model_audit: dict | None = None,
) -> str:
    audit = model_audit or {}
    model_family = audit.get("model_family", "RandomForest")
    feature_count = audit.get("feature_count", 50)
    data_source = audit.get("data_source", "Binance live 1m K 線")
    pipeline = audit.get("pipeline_source", "live_feature_pipeline")
    latest_kline = audit.get("latest_candle_time_utc", "N/A")
    feature_ts = audit.get("feature_timestamp_utc", "N/A")
    feat_hash = audit.get("feature_hash", "")
    model_load_status = "正常" if audit.get("model_loaded") else "異常"

    lines = [
        "📊 <b>CSFFM Bot 狀態</b>",
        "Bot 狀態：在線",
        f"模型：{safe_text(model_family)}",
        f"特徵數：{safe_text(feature_count)}",
        f"資料來源：{safe_text(data_source)}",
        f"特徵管線：{safe_text(pipeline)}",
        f"最新 K 線：{safe_text(latest_kline)} UTC",
        f"特徵末筆：{safe_text(feature_ts)} UTC",
        f"特徵 hash：{safe_text(feat_hash) if feat_hash else '—'}",
        f"模型載入：{model_load_status}",
        f"Scheduler：{'啟用中' if scheduler_running else '未啟動（缺 TELEGRAM_CHAT_ID 或未排程）'}",
        f"推論 regime：{safe_text(PREDICTION_REGIME)}",
        f"上次推播：{safe_text(last_push or '（尚無）')}",
        f"上次觸發：{safe_text(last_trigger or '—')}",
    ]
    if snap:
        if snap.stale:
            lines.append(f"資料：⚠️ {safe_text(snap.stale_reason)}")
        else:
            lines.append(
                f"資料：K 線延遲 {snap.kline_delay_seconds:.0f}s｜特徵 {snap.feature_timestamp}"
            )
    return _finalize_message("\n".join(lines))