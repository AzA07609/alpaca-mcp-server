"""Read-only TFW 1W + TFD 1D trading strategy analysis."""

from __future__ import annotations

import traceback
from typing import Any

import httpx
from fastmcp import FastMCP

from .fvp import _fetch_daily_bars, _profile


def _ema(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    alpha = 2.0 / (period + 1.0)
    prev = seed
    for i in range(period, len(values)):
        prev = (values[i] - prev) * alpha + prev
        out[i] = prev
    return out


def _sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    total = sum(values[:period])
    out[period - 1] = total / period
    for i in range(period, len(values)):
        total += values[i] - values[i - period]
        out[i] = total / period
    return out


def _cci(bars: list[dict[str, Any]], period: int = 55) -> list[float | None]:
    tp = [(float(b["h"]) + float(b["l"]) + float(b["c"])) / 3.0 for b in bars]
    out: list[float | None] = [None] * len(tp)
    if len(tp) < period:
        return out
    for i in range(period - 1, len(tp)):
        window = tp[i - period + 1 : i + 1]
        mean = sum(window) / period
        dev = sum(abs(x - mean) for x in window) / period
        out[i] = 0.0 if dev == 0 else (tp[i] - mean) / (0.015 * dev)
    return out


def _stoch(bars: list[dict[str, Any]], period: int = 5, smooth: int = 3) -> tuple[float | None, float | None]:
    k_values: list[float] = []
    for i in range(period - 1, len(bars)):
        window = bars[i - period + 1 : i + 1]
        high = max(float(x["h"]) for x in window)
        low = min(float(x["l"]) for x in window)
        close = float(bars[i]["c"])
        k_values.append(50.0 if high == low else (close - low) / (high - low) * 100.0)
    if not k_values:
        return None, None
    k = sum(k_values[-smooth:]) / min(smooth, len(k_values))
    d_window = k_values[-smooth * 2 :]
    d = sum(d_window[-smooth:]) / min(smooth, len(d_window))
    return k, d


def _latest_daily_indicators(bars: list[dict[str, Any]]) -> dict[str, Any]:
    closes = [float(b["c"]) for b in bars]
    volumes = [float(b["v"]) for b in bars]
    sma200 = _sma(closes, 200)
    cci55 = _cci(bars, 55)
    cci_sma14 = _sma([x if x is not None else 0.0 for x in cci55], 14)
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line: list[float] = []
    for i in range(len(closes)):
        if ema12[i] is not None and ema26[i] is not None:
            macd_line.append(float(ema12[i]) - float(ema26[i]))
        else:
            macd_line.append(0.0)
    signal = _ema(macd_line, 9)
    hist = [macd_line[i] - float(signal[i]) if signal[i] is not None else 0.0 for i in range(len(closes))]
    bb_mid = _sma(closes, 20)
    bb_upper: list[float | None] = [None] * len(closes)
    bb_lower: list[float | None] = [None] * len(closes)
    for i in range(19, len(closes)):
        w = closes[i - 19 : i + 1]
        mean = sum(w) / 20.0
        variance = sum((x - mean) ** 2 for x in w) / 20.0
        sd = variance ** 0.5
        bb_upper[i] = mean + 2.0 * sd
        bb_lower[i] = mean - 2.0 * sd
    stoch_k, stoch_d = _stoch(bars)
    avg20_volume = sum(volumes[-20:]) / min(20, len(volumes))
    recent_high = max(float(b["h"]) for b in bars[-20:])
    recent_low = min(float(b["l"]) for b in bars[-20:])
    last = bars[-1]
    prev = bars[-2] if len(bars) > 1 else last
    return {
        "price": float(last["c"]),
        "date": last.get("t"),
        "sma200": sma200[-1],
        "cci55": cci55[-1],
        "cci55_sma14": cci_sma14[-1],
        "macd": macd_line[-1],
        "macd_signal": signal[-1],
        "macd_histogram": hist[-1],
        "macd_histogram_previous": hist[-2] if len(hist) > 1 else None,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "bb_mid": bb_mid[-1],
        "bb_upper": bb_upper[-1],
        "bb_lower": bb_lower[-1],
        "volume": volumes[-1],
        "volume_avg20": avg20_volume,
        "volume_ratio": volumes[-1] / avg20_volume if avg20_volume else None,
        "recent_20d_high": recent_high,
        "recent_20d_low": recent_low,
        "previous_close": float(prev["c"]),
    }


def _weekly_trend(bars: list[dict[str, Any]]) -> dict[str, Any]:
    closes = [float(b["c"]) for b in bars]
    sma200 = _sma(closes, 200)
    last = closes[-1]
    above = sma200[-1] is not None and last > float(sma200[-1])
    higher_high = len(closes) >= 12 and max(closes[-4:]) > max(closes[-12:-4])
    higher_low = len(closes) >= 12 and min(closes[-4:]) > min(closes[-12:-4])
    if above and higher_high and higher_low:
        trend = "BULLISH"
    elif not above and not higher_high and not higher_low:
        trend = "BEARISH"
    else:
        trend = "MIXED"
    return {"trend": trend, "price": last, "sma200": sma200[-1], "higher_high": higher_high, "higher_low": higher_low}


def _round(v: Any) -> Any:
    return round(float(v), 4) if isinstance(v, (int, float)) else v


def _decision(weekly: dict[str, Any], daily: dict[str, Any], fvp: dict[str, Any]) -> dict[str, Any]:
    price = daily["price"]
    vah, poc, val = fvp.get("vah"), fvp.get("poc"), fvp.get("val")
    hist = daily["macd_histogram"]
    prev_hist = daily["macd_histogram_previous"]
    volume_ok = daily["volume_ratio"] is not None and daily["volume_ratio"] >= 1.0
    near_vah = vah is not None and abs(price - vah) / price <= 0.025
    near_poc = poc is not None and abs(price - poc) / price <= 0.025
    near_val = val is not None and abs(price - val) / price <= 0.04
    bullish_momentum = hist is not None and prev_hist is not None and hist > prev_hist
    above_sma = daily["sma200"] is not None and price > daily["sma200"]
    if weekly["trend"] == "BULLISH" and above_sma and (near_vah or near_poc or near_val) and bullish_momentum:
        action = "WATCH_LONG"
    else:
        action = "WAIT"
    location = "ABOVE_VALUE_AREA"
    if near_vah:
        location = "NEAR_VAH"
    elif near_poc:
        location = "NEAR_POC"
    elif near_val:
        location = "NEAR_VAL"
    return {
        "action": action,
        "location": location,
        "weekly_filter": weekly["trend"],
        "confirmation": {
            "above_daily_sma200": above_sma,
            "macd_histogram_rising": bullish_momentum,
            "volume_at_or_above_avg20": volume_ok,
        },
        "rule": "No immediate BUY when price is outside the calculated pullback zone; prefer R:R >= 1:2.",
    }


def register_strategy_analysis_tool(server: FastMCP, client: httpx.AsyncClient) -> None:
    """Register the read-only combined strategy analysis tool."""

    @server.tool(
        annotations={
            "title": "Analyze TFW 1W + TFD 1D Strategy",
            "readOnlyHint": True,
            "openWorldHint": True,
        }
    )
    async def analyze_tfw_tfd_strategy(
        symbol: str,
        daily_lookback: int = 220,
        weekly_lookback: int = 220,
        bins: int = 48,
        feed: str | None = "iex",
    ) -> dict[str, Any]:
        """Analyze locked Strategy v1: Weekly trend + Daily setup + FVP location."""
        try:
            symbol = symbol.upper().strip()
            if not symbol:
                return {"error": {"message": "symbol is required"}}

            # Daily helper now returns the most recent bars in chronological order.
            daily_data = await _fetch_daily_bars(client, symbol, daily_lookback, feed)
            if "error" in daily_data:
                return daily_data
            daily_bars = daily_data.get("bars", [])
            if len(daily_bars) < 200:
                return {"error": {"message": "At least 200 daily bars are required for SMA200.", "daily_bars": len(daily_bars)}}

            # Weekly endpoint: request from a broad historical start, but use
            # descending order so `limit` selects the LATEST weekly bars.
            path = f"/v2/stocks/{symbol}/bars"
            response = await client.get(
                path,
                params={
                    "timeframe": "1Week",
                    "start": "2021-01-01T00:00:00Z",
                    "limit": min(max(int(weekly_lookback), 200), 1000),
                    "sort": "desc",
                    "adjustment": "raw",
                    "feed": (feed or "iex").strip().lower(),
                },
            )
            response.raise_for_status()
            weekly_json = response.json()
            weekly_bars = weekly_json.get("bars", []) if isinstance(weekly_json, dict) else []
            if not isinstance(weekly_bars, list):
                weekly_bars = []
            weekly_bars.reverse()
            if len(weekly_bars) < 200:
                return {"error": {"message": "Not enough weekly bars for SMA200", "weekly_bars": len(weekly_bars)}}

            daily = _latest_daily_indicators(daily_bars)
            weekly = _weekly_trend(weekly_bars)
            # FVP is a TFD location layer: use the same latest daily window.
            fvp = _profile(daily_bars[-min(60, len(daily_bars)):], bins)
            if "error" in fvp:
                return fvp
            decision = _decision(weekly, daily, fvp)
            return {
                "symbol": symbol,
                "strategy": "Strategy v1 — TFW 1Week + TFD 1Day",
                "timeframes": {"TFW": "1Week", "TFD": "1Day"},
                "decision": decision,
                "TFW": {k: _round(v) for k, v in weekly.items()},
                "TFD": {k: _round(v) for k, v in daily.items()},
                "FVP": {k: _round(v) for k, v in fvp.items() if k != "profile"},
                "FVP_high_volume_nodes": fvp.get("high_volume_nodes", []),
                "method": "Daily OHLCV typical-price allocation; latest 60 daily bars; 70% value area; read-only.",
                "data": {
                    "daily_bars": len(daily_bars),
                    "weekly_bars": len(weekly_bars),
                    "daily_first_bar": daily_bars[0].get("t") if daily_bars else None,
                    "daily_last_bar": daily_bars[-1].get("t") if daily_bars else None,
                    "weekly_first_bar": weekly_bars[0].get("t") if weekly_bars else None,
                    "weekly_last_bar": weekly_bars[-1].get("t") if weekly_bars else None,
                    "feed": (feed or "iex").strip().lower(),
                },
            }
        except Exception as exc:
            return {"error": {"message": "Strategy analysis failed", "error_type": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc(limit=12)}}
