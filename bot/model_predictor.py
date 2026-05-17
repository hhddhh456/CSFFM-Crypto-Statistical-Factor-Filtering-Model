# -*- coding: utf-8 -*-
"""
Phase 6 模型推論：扁平 formatter dict、daily_realized_vol、動態履約價（§10）。
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtest.evaluation_utils import LABEL_ORDER
from backtest.window_analysis import extract_window_from_feature
from bot.config import (
    BACKTEST_ROOT,
    DAILY_VOL_MAX,
    DIRECTIONAL_THRESHOLD,
    FRESHNESS_MAX_SECONDS,
    HIGH_CONVICTION_THRESHOLD,
    LABELS_FOR_REPORT,
    PREDICTION_REGIME,
    PROBA_HIGH_CONF,
    TAIWAN_TZ,
    NY_TZ,
    UTC_TZ,
    TAIL_RISK_HIGH,
    strike_tick,
)
from bot.feature_consistency import get_model_feature_names, validate_feature_consistency
from bot.live_feature_pipeline import (
    LiveSnapshot,
    build_live_snapshot_or_raise,
    check_market_data_freshness,
)
from bot.oos_eligibility import is_label_eligible
from models.model_utils import Phase4Paths, load_model
from models.regime_utils import regime_folder_name

logger = logging.getLogger(__name__)
_PHASE4 = Phase4Paths.default()
_REFERENCE_SYMBOL = "BTCUSDT"
_REFERENCE_LABEL = LABELS_FOR_REPORT[0]


def _floor_to_tick(value: float, tick: int) -> int:
    return int(math.floor(value / tick) * tick)


def _ceil_to_tick(value: float, tick: int) -> int:
    return int(math.ceil(value / tick) * tick)


def signal_confidence(proba: float) -> float:
    return float(max(proba, 1.0 - proba))


def _signed_proba_delta(proba: float) -> float:
    """將機率映射為 [-1, 1] 方向強度。"""
    if proba != proba:
        return 0.0
    return float((proba - 0.5) * 2.0)


def _format_vol_magnitude(
    *,
    daily_vol: float,
    proba: float,
    strike: dict[str, Any] | None,
) -> str:
    if strike and strike.get("vol_range_pct") is not None:
        pct = float(strike["vol_range_pct"])
    elif daily_vol == daily_vol and daily_vol > 0:
        pct = round(daily_vol * 100, 2)
    else:
        return "N/A"
    sign = 1.0 if _signed_proba_delta(proba) >= 0 else -1.0
    return f"{sign * pct:+.2f}%"


def _format_risk_magnitude(proba: float) -> str:
    delta = _signed_proba_delta(proba)
    if delta == 0.0:
        return "0.0"
    return f"{delta * 10:+.1f}"


def predict_label_proba(X_row: pd.DataFrame, symbol: str, label_type: str) -> float:
    model = load_model(symbol, label_type, regime=PREDICTION_REGIME)
    return float(model.predict_proba(X_row)[:, 1][0])


def _label_direction_text(label_type: str, proba: float, *, eligible: bool) -> tuple[str, int]:
    conf_pct = int(signal_confidence(proba) * 100)
    if not eligible:
        return "（OOS 未達標）", conf_pct
    if label_type == "label_realized_volatility":
        if proba >= PROBA_HIGH_CONF:
            return "擴大", conf_pct
        if proba <= 1 - PROBA_HIGH_CONF:
            return "收斂", conf_pct
        return "震盪/不明", conf_pct
    if label_type in ("label_kurtosis", "label_jarque_bera"):
        if proba >= PROBA_HIGH_CONF:
            return "升高", conf_pct
        if proba <= 1 - PROBA_HIGH_CONF:
            return "降低", conf_pct
        return "中性", conf_pct
    if label_type == "label_sequential_correlation":
        if proba >= PROBA_HIGH_CONF:
            return "正相關偏高", conf_pct
        if proba <= 1 - PROBA_HIGH_CONF:
            return "正相關偏低", conf_pct
        return "不明", conf_pct
    return "—", conf_pct


def _seller_direction_from_seq(proba: float) -> tuple[str, float]:
    conf = signal_confidence(proba)
    if proba >= PROBA_HIGH_CONF:
        return "bullish", conf
    if proba <= 1 - PROBA_HIGH_CONF:
        return "bearish", conf
    return "neutral", conf


def calculate_recommended_strike(
    symbol: str,
    current_price: float,
    daily_realized_vol: float,
    direction: str,
    direction_conf: float,
    kurtosis_high: bool,
    historical_avg_vol: Optional[float] = None,
    risk_aversion: str = "medium",
    tail_risk_score: Optional[float] = None,
    directional_threshold: float = DIRECTIONAL_THRESHOLD,
) -> dict[str, Any]:
    """§10 賣方履約價（24H 持倉、floor/ceil tick）。"""
    symbol = symbol.upper().strip().replace("USDT", "")

    if current_price <= 0 or np.isnan(current_price):
        raise ValueError("current_price 必須為正數")
    if daily_realized_vol <= 0 or np.isnan(daily_realized_vol):
        raise ValueError("daily_realized_vol 必須為正數且不能為 NaN")
    if daily_realized_vol >= DAILY_VOL_MAX:
        raise ValueError("daily_realized_vol 過高，疑似 annualized vol 或模型異常")

    multiplier_map = {"low": 1.6, "medium": 2.0, "high": 2.8}
    base_multiplier = multiplier_map.get(risk_aversion, 2.0)

    if tail_risk_score is not None:
        tail_risk_score = max(0.0, min(float(tail_risk_score), 1.0))
        base_multiplier *= 1.0 + 0.8 * tail_risk_score
    elif kurtosis_high:
        base_multiplier += 0.8

    offset = current_price * daily_realized_vol * base_multiplier
    tick = strike_tick(symbol)

    if historical_avg_vol is not None and historical_avg_vol > 0:
        vol_ratio = daily_realized_vol / historical_avg_vol
        if vol_ratio >= 1.45:
            vol_strength = "明顯擴大（高於近期均值 45%+）"
        elif vol_ratio >= 1.20:
            vol_strength = "中性偏高"
        elif vol_ratio >= 0.85:
            vol_strength = "相對平穩"
        else:
            vol_strength = "明顯收斂"
    else:
        high_th = 0.028 if symbol == "BTC" else 0.035
        mid_th = 0.019 if symbol == "BTC" else 0.024
        if daily_realized_vol >= high_th:
            vol_strength = "明顯擴大"
        elif daily_realized_vol >= mid_th:
            vol_strength = "中性偏高"
        elif daily_realized_vol >= mid_th * 0.75:
            vol_strength = "相對平穩"
        else:
            vol_strength = "明顯收斂"

    if direction == "bullish" and direction_conf >= directional_threshold:
        raw_put = current_price - offset * 0.95
        raw_call = current_price + offset * 1.30
        put_strike = _floor_to_tick(raw_put, tick)
        call_strike = _ceil_to_tick(raw_call, tick)
        recommended_side = "Short PUT（看漲環境賣 Put）"
        final_multiplier = base_multiplier * 0.95
        suggested_price_note = f"核心建議價 PUT：{put_strike}"
    elif direction == "bearish" and direction_conf >= directional_threshold:
        raw_call = current_price + offset * 0.95
        raw_put = current_price - offset * 1.30
        call_strike = _ceil_to_tick(raw_call, tick)
        put_strike = _floor_to_tick(raw_put, tick)
        recommended_side = "Short CALL（看跌環境賣 Call）"
        final_multiplier = base_multiplier * 0.95
        suggested_price_note = f"核心建議價 CALL：{call_strike}"
    else:
        raw_put = current_price - offset * 1.10
        raw_call = current_price + offset * 1.10
        put_strike = _floor_to_tick(raw_put, tick)
        call_strike = _ceil_to_tick(raw_call, tick)
        recommended_side = "Short STRANGLE / IRON CONDOR（中性）"
        final_multiplier = base_multiplier * 1.10
        suggested_price_note = f"雙向建議 PUT：{put_strike}｜CALL：{call_strike}"

    if kurtosis_high or (tail_risk_score is not None and tail_risk_score >= TAIL_RISK_HIGH):
        risk_structure_note = "極端風險偏高，建議優先使用 defined-risk spread 或降低倉位"
    else:
        risk_structure_note = "波動可控，可依倉位控管執行賣方策略"

    if final_multiplier >= 2.5:
        otm_level = "Deep OTM（深價外）"
    elif final_multiplier >= 1.8:
        otm_level = "OTM（價外）"
    else:
        otm_level = "Near ATM（近平值）"

    extreme_risk = "顯著提升（肥尾風險高）" if kurtosis_high else "處於正常區間"
    risk_note = (
        "注意突發極端行情，建議嚴格控管部位"
        if kurtosis_high
        else "波動可控，但仍需設定停損與倉位上限"
    )

    expected_min = current_price * (1 - daily_realized_vol)
    expected_max = current_price * (1 + daily_realized_vol)
    vol_range_pct = round(daily_realized_vol * 100, 2)

    return {
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "daily_realized_vol": round(daily_realized_vol, 6),
        "recommended_side": recommended_side,
        "put_strike": put_strike,
        "call_strike": call_strike,
        "otm_level": otm_level,
        "vol_strength": vol_strength,
        "extreme_risk": extreme_risk,
        "risk_note": risk_note,
        "risk_structure_note": risk_structure_note,
        "suggested_price_note": suggested_price_note,
        "expected_min": round(expected_min, 2),
        "expected_max": round(expected_max, 2),
        "vol_range_pct": vol_range_pct,
        "vol_range_str": f"±{vol_range_pct}%",
    }


def load_top_mda_factors(*, top_n: int = 5) -> str:
    sym = "BTCUSDT"
    folder = regime_folder_name(PREDICTION_REGIME)
    lines: list[str] = []
    for label in LABELS_FOR_REPORT:
        p = BACKTEST_ROOT / sym / folder / f"mda_{label}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        if "importance_mean" not in df.columns:
            continue
        for _, r in df.sort_values("importance_mean", ascending=False).head(3).iterrows():
            feat = str(r.get("feature", ""))
            w = extract_window_from_feature(feat)
            lines.append(f"{feat} ({w}分)" if w else feat)
    seen: set[str] = set()
    out: list[str] = []
    for x in lines:
        if x not in seen:
            seen.add(x)
            out.append(x)
        if len(out) >= top_n:
            break
    return "\n".join(f"• {x}" for x in out) if out else "• （無 MDA 資料）"


def _seller_bias_level(dir_conf: float) -> str:
    if dir_conf >= HIGH_CONVICTION_THRESHOLD:
        return "高信心"
    if dir_conf >= DIRECTIONAL_THRESHOLD:
        return "方向明確"
    return "中性"


def _build_direction_advice(
    labels: dict[str, dict],
    seller_dir: str,
    dir_conf: float,
) -> str:
    vol = labels.get("label_realized_volatility", {})
    kurt = labels.get("label_kurtosis", {})
    jb = labels.get("label_jarque_bera", {})
    seq = labels.get("label_sequential_correlation", {})
    conv = _seller_bias_level(dir_conf)
    lines = [
        f"• 波動（Volatility）：{vol.get('direction', 'N/A')}（信心 {vol.get('confidence_pct', 'N/A')}%）",
        (
            f"• 極端風險：峰度（Kurtosis）：{kurt.get('direction', 'N/A')}"
            f"｜Jarque-Bera 檢定：{jb.get('direction', 'N/A')}"
        ),
        (
            f"• 序列相關（Sequential Correlation）：{seq.get('direction', 'N/A')}"
            f"（信心 {seq.get('confidence_pct', 'N/A')}%）"
        ),
        f"• 賣方輔助：{seller_dir}（{int(dir_conf * 100)}%，{conv}）",
    ]
    return "\n".join(lines)


def _predict_symbol_flat(
    prefix: str,
    sym: str,
    snap,
    labels: dict,
    strike: dict | None,
    probas: dict[str, float],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pfx = prefix
    price = snap.prices.get(sym, float("nan"))
    if strike:
        out[f"{pfx}_current_price"] = strike.get("current_price", price)
    elif price == price:
        out[f"{pfx}_current_price"] = round(price, 2)
    else:
        out[f"{pfx}_current_price"] = "N/A"

    for label in LABELS_FOR_REPORT:
        key = label.replace("label_", "")
        if label == "label_realized_volatility":
            short = "vol"
        elif label == "label_jarque_bera":
            short = "jb"
        elif label == "label_kurtosis":
            short = "kurt"
        else:
            short = "seq"
        d = labels.get(label, {})
        out[f"{pfx}_{short}_dir"] = d.get("direction", "—")
        out[f"{pfx}_{short}_prob"] = d.get("confidence_pct", "N/A")

    rv_p = probas.get("label_realized_volatility", 0.5)
    kurt_p = probas.get("label_kurtosis", 0.5)
    jb_p = probas.get("label_jarque_bera", 0.5)
    daily_vol = snap.current_rv.get(sym, float("nan"))
    out[f"{pfx}_vol_magnitude"] = _format_vol_magnitude(
        daily_vol=daily_vol,
        proba=rv_p if rv_p == rv_p else 0.5,
        strike=strike,
    )
    out[f"{pfx}_kurt_magnitude"] = _format_risk_magnitude(kurt_p if kurt_p == kurt_p else 0.5)
    out[f"{pfx}_jb_magnitude"] = _format_risk_magnitude(jb_p if jb_p == jb_p else 0.5)

    seq_p = probas.get("label_sequential_correlation", 0.5)
    seller_dir, dir_conf = _seller_direction_from_seq(seq_p if seq_p == seq_p else 0.5)
    out[f"{pfx}_seller_bias"] = seller_dir
    out[f"{pfx}_seller_bias_prob"] = int(dir_conf * 100)
    out[f"{pfx}_seller_bias_level"] = _seller_bias_level(dir_conf)

    if strike:
        out[f"{pfx}_vol_range_str"] = strike.get("vol_range_str", "N/A")
        out[f"{pfx}_expected_min"] = strike.get("expected_min", "N/A")
        out[f"{pfx}_expected_max"] = strike.get("expected_max", "N/A")
        out[f"{pfx}_direction_advice"] = labels.get("_direction_advice", "—")
        out[f"{pfx}_recommended_side"] = strike.get("recommended_side", "暫停")
        out[f"{pfx}_otm_level"] = strike.get("otm_level", "N/A")
        out[f"{pfx}_put_strike"] = strike.get("put_strike", "N/A")
        out[f"{pfx}_call_strike"] = strike.get("call_strike", "N/A")
        out[f"{pfx}_suggested_price_note"] = strike.get("suggested_price_note", "")
        out[f"{pfx}_vol_strength"] = strike.get("vol_strength", "N/A")
        out[f"{pfx}_extreme_risk"] = strike.get("extreme_risk", "N/A")
        out[f"{pfx}_risk_note"] = strike.get("risk_note", "N/A")
        out[f"{pfx}_risk_structure_note"] = strike.get("risk_structure_note", "N/A")
    else:
        for k in (
            "vol_range_str", "expected_min", "expected_max", "direction_advice",
            "recommended_side", "otm_level", "put_strike", "call_strike",
            "suggested_price_note", "vol_strength", "extreme_risk", "risk_note",
            "risk_structure_note", "seq_dir", "seq_prob",
            "seller_bias", "seller_bias_prob", "seller_bias_level",
        ):
            out[f"{pfx}_{k}"] = "暫停"
        out[f"{pfx}_direction_advice"] = "資料延遲，暫停方向性與期權建議。"
        out[f"{pfx}_seller_bias"] = "N/A"
        out[f"{pfx}_seller_bias_prob"] = "N/A"
        out[f"{pfx}_seller_bias_level"] = "N/A"
        out[f"{pfx}_vol_magnitude"] = "N/A"
        out[f"{pfx}_kurt_magnitude"] = "N/A"
        out[f"{pfx}_jb_magnitude"] = "N/A"
    return out


def _run_symbol(sym: str, snap) -> tuple[dict, dict | None, dict[str, float]]:
    X = snap.X_row
    labels: dict[str, dict] = {}
    probas: dict[str, float] = {}

    for label in LABEL_ORDER:
        try:
            probas[label] = predict_label_proba(X, sym, label)
        except FileNotFoundError:
            probas[label] = float("nan")

    for label in LABELS_FOR_REPORT:
        p = probas.get(label, float("nan"))
        if p != p:
            p = 0.5
        elig = is_label_eligible(sym, PREDICTION_REGIME, label)
        txt, conf = _label_direction_text(label, p, eligible=elig)
        labels[label] = {"direction": txt, "confidence_pct": conf, "eligible": elig}

    rv_p = probas.get("label_realized_volatility", 0.5)
    kurt_p = probas.get("label_kurtosis", 0.5)
    jb_p = probas.get("label_jarque_bera", 0.5)
    seq_p = probas.get("label_sequential_correlation", 0.5)

    tail_risk = max(
        signal_confidence(kurt_p) if kurt_p == kurt_p else 0,
        signal_confidence(jb_p) if jb_p == jb_p else 0,
    )
    kurt_high = (
        signal_confidence(kurt_p) >= PROBA_HIGH_CONF and kurt_p >= PROBA_HIGH_CONF and is_label_eligible(
            sym, PREDICTION_REGIME, "label_kurtosis"
        )
    ) or (
        signal_confidence(jb_p) >= PROBA_HIGH_CONF
        and jb_p >= PROBA_HIGH_CONF
        and is_label_eligible(sym, PREDICTION_REGIME, "label_jarque_bera")
    )

    seller_dir, dir_conf = _seller_direction_from_seq(seq_p if seq_p == seq_p else 0.5)
    labels["_direction_advice"] = _build_direction_advice(labels, seller_dir, dir_conf)

    price = snap.prices.get(sym, float("nan"))
    daily_vol = snap.current_rv.get(sym, float("nan"))
    hist = snap.historical_avg_vol.get(sym)

    strike = None
    if not snap.stale and price == price and daily_vol == daily_vol and daily_vol > 0:
        strike = calculate_recommended_strike(
            symbol=sym,
            current_price=price,
            daily_realized_vol=daily_vol,
            direction=seller_dir,
            direction_conf=dir_conf,
            kurtosis_high=kurt_high,
            historical_avg_vol=hist if hist == hist else None,
            tail_risk_score=tail_risk,
        )

    return labels, strike, probas


def _vpin_snapshot_values(X_row: pd.DataFrame) -> dict[str, float]:
    """從 live 特徵列擷取 VPIN（優先 50 分鐘窗口）。"""
    out: dict[str, float] = {}
    for sym_prefix, key in (("btc_", "BTC"), ("eth_", "ETH")):
        preferred = f"{sym_prefix}vpin_50"
        if preferred in X_row.columns:
            val = float(X_row[preferred].iloc[0])
        else:
            cols = [c for c in X_row.columns if c.startswith(sym_prefix) and "vpin" in c.lower()]
            if not cols:
                continue
            val = float(X_row[cols[0]].iloc[0])
        if val == val:
            out[key] = round(val, 4)
    return out


def build_model_audit(
    snap: LiveSnapshot,
    *,
    feature_consistency: dict[str, Any],
    model_loaded: bool,
) -> dict[str, Any]:
    model_path = str(
        _PHASE4.model_path(_REFERENCE_SYMBOL, PREDICTION_REGIME, _REFERENCE_LABEL)
    )
    feature_list_path = str(_PHASE4.features_combined_path("BTCUSDT"))
    feature_count = len(snap.X_row.columns)
    return {
        "model_family": "RandomForest",
        "model_path": model_path,
        "feature_list_path": feature_list_path,
        "feature_count": feature_count,
        "pipeline_source": "live_feature_pipeline",
        "prediction_source": "rf_predict_proba",
        "model_loaded": model_loaded,
        "feature_pipeline_ok": bool(feature_consistency.get("ok")),
        "latest_candle_time_utc": snap.latest_kline_time.strftime("%Y-%m-%d %H:%M:%S"),
        "vpin_enabled": bool(feature_consistency.get("vpin_enabled")),
    }


def get_model_status_audit(snap: LiveSnapshot | None = None) -> dict[str, Any]:
    """供 /status 使用；不執行完整推論。"""
    model_loaded = False
    feature_count = 50
    feature_pipeline_ok = False
    latest_kline = "N/A"
    vpin_enabled = False
    names: list[str] = []
    try:
        load_model(_REFERENCE_SYMBOL, _REFERENCE_LABEL, regime=PREDICTION_REGIME)
        model_loaded = True
        names = get_model_feature_names(
            _REFERENCE_SYMBOL, _REFERENCE_LABEL, regime=PREDICTION_REGIME
        )
        if names:
            feature_count = len(names)
    except FileNotFoundError:
        model_loaded = False

    if snap is not None:
        latest_kline = snap.latest_kline_time.strftime("%Y-%m-%d %H:%M:%S")
        if model_loaded and names:
            consistency = validate_feature_consistency(names, list(snap.X_row.columns))
            feature_pipeline_ok = consistency["ok"]
            vpin_enabled = consistency["vpin_enabled"]
        elif snap.X_row is not None and not snap.X_row.empty:
            feature_count = len(snap.X_row.columns)
            feature_pipeline_ok = snap.X_row.shape[1] == 50

    return {
        "model_family": "RandomForest",
        "model_path": str(
            _PHASE4.model_path(_REFERENCE_SYMBOL, PREDICTION_REGIME, _REFERENCE_LABEL)
        ),
        "feature_list_path": str(_PHASE4.features_combined_path("BTCUSDT")),
        "feature_count": feature_count,
        "pipeline_source": "live_feature_pipeline",
        "data_source": "Binance live 1m K 線",
        "latest_candle_time_utc": latest_kline,
        "model_loaded": model_loaded,
        "feature_pipeline_ok": feature_pipeline_ok,
        "vpin_enabled": vpin_enabled,
    }


def get_full_prediction() -> dict[str, Any]:
    """產生完整扁平 dict，供 format_telegram_message 使用。"""
    snap = build_live_snapshot_or_raise()
    check_market_data_freshness(snap)

    model_loaded = True
    try:
        model_features = get_model_feature_names(
            _REFERENCE_SYMBOL, _REFERENCE_LABEL, regime=PREDICTION_REGIME
        )
        load_model(_REFERENCE_SYMBOL, _REFERENCE_LABEL, regime=PREDICTION_REGIME)
    except FileNotFoundError:
        model_loaded = False
        model_features = list(snap.X_row.columns)

    live_features = list(snap.X_row.columns)
    feature_consistency = validate_feature_consistency(model_features, live_features)
    logger.info("Feature consistency check: %s", feature_consistency)

    if not feature_consistency["ok"]:
        missing = feature_consistency["missing_in_live"]
        raise RuntimeError(
            f"即時特徵與模型不一致，缺少 {len(missing)} 欄：{missing[:5]}"
            + ("…" if len(missing) > 5 else "")
        )

    model_audit = build_model_audit(
        snap, feature_consistency=feature_consistency, model_loaded=model_loaded
    )
    logger.info("Model audit: %s", model_audit)

    now_utc = datetime.now(ZoneInfo(UTC_TZ))
    now_tw = now_utc.astimezone(ZoneInfo(TAIWAN_TZ))
    now_ny = now_utc.astimezone(ZoneInfo(NY_TZ))

    btc_labels, btc_strike, btc_probas = _run_symbol("BTCUSDT", snap)
    eth_labels, eth_strike, eth_probas = _run_symbol("ETHUSDT", snap)

    data: dict[str, Any] = {
        "taiwan_time": now_tw.strftime("%Y-%m-%d %H:%M:%S"),
        "utc_time": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "ny_time": now_ny.strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": "Multi-timeframe RF",
        "data_freshness_note": (
            f"Binance 即時 1m K 線，末筆 {snap.latest_kline_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        ),
    }

    data.update(
        _predict_symbol_flat("btc", "BTCUSDT", snap, btc_labels, btc_strike, btc_probas)
    )
    data.update(
        _predict_symbol_flat("eth", "ETHUSDT", snap, eth_labels, eth_strike, eth_probas)
    )

    data["_model_audit"] = model_audit
    data["_feature_consistency"] = feature_consistency

    if feature_consistency.get("vpin_enabled"):
        vpin_vals = _vpin_snapshot_values(snap.X_row)
        if vpin_vals:
            parts = [f"{k} VPIN(50)={v}" for k, v in vpin_vals.items()]
            data["vpin_report_line"] = "微結構 VPIN：" + "｜".join(parts)

    return data
