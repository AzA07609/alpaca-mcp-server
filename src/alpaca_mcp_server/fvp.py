"""Fixed Range Volume Profile analysis for stock data.

This is a read-only, daily-bar approximation of a TradingView Fixed Range
Volume Profile. It distributes each bar's volume to the price bin containing
its typical price (H+L+C)/3, then calculates POC and a 70% value area.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastmcp import FastMCP


async def _fetch_daily_bars(
    client: httpx.AsyncClient, symbol: str, lookback: int, feed: str | None
) -> dict[str, Any]:
    """Fetch enough calendar days to obtain the requested daily bars.

    The Alpaca historical-bars API defaults ``start`` to the beginning of the
    current day when it is omitted. On weekends/holidays that produces an
    empty result, even for valid symbols. FVP needs an explicit historical
    range based on the requested lookback.

    The single-symbol endpoint is used deliberately because its response is
    always ``{"bars": [...]}``, avoiding the multi-symbol response shape
    ``{"bars": {"AAPL": [...]}}``.
    """
    selected_feed = (feed or "iex").strip().lower()
    # Daily bars occur only on trading days. Two calendar days per requested
    # bar gives comfortable room for weekends and market holidays.
    calendar_days = max(int(lookback) * 2, int(lookback) + 10)
    start = datetime.now(timezone.utc) - timedelta(days=calendar_days)
    start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "timeframe": "1Day",
        "start": start_iso,
        "limit": min(max(int(lookback), 10), 1000),
        "sort": "asc",
        "adjustment": "raw",
        "feed": selected_feed,
    }
    try:
        response = await client.get(
            f"/v2/stocks/{symbol.upper().strip()}/bars", params=params
        )
        if response.is_error:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            return {
                "error": {
                    "message": "Alpaca market data request failed",
                    "http_status": response.status_code,
                    "detail": detail,
                    "feed_used": selected_feed,
                    "request_start": start_iso,
                }
            }
        try:
            data = response.json()
        except ValueError:
            return {
                "error": {
                    "message": "Alpaca returned a non-JSON response",
                    "http_status": response.status_code,
                    "feed_used": selected_feed,
                    "request_start": start_iso,
                }
            }

        # Single-symbol endpoint returns a list. Keep a defensive fallback for
        # the multi-symbol shape in case an upstream proxy changes the route.
        bars = data.get("bars", []) if isinstance(data, dict) else []
        if isinstance(bars, dict):
            bars = bars.get(symbol.upper().strip(), [])
        if not isinstance(bars, list):
            bars = []
        return {
            "bars": bars,
            "feed_used": selected_feed,
            "request_start": start_iso,
        }
    except httpx.HTTPError as exc:
        return {
            "error": {
                "message": f"Market data request failed: {exc}",
                "feed_used": selected_feed,
                "request_start": start_iso,
            }
        }


def _profile(bars: list[dict[str, Any]], bins: int) -> dict[str, Any]:
    cleaned: list[dict[str, float]] = []
    for bar in bars:
        try:
            high = float(bar["h"])
            low = float(bar["l"])
            close = float(bar["c"])
            volume = float(bar["v"])
        except (KeyError, TypeError, ValueError):
            continue
        if high >= low and volume >= 0:
            cleaned.append({"high": high, "low": low, "close": close, "volume": volume})

    if not cleaned:
        return {"error": {"message": "No usable OHLCV bars were returned."}}

    low = min(b["low"] for b in cleaned)
    high = max(b["high"] for b in cleaned)
    if high == low:
        return {
            "bars_used": len(cleaned),
            "range_low": low,
            "range_high": high,
            "poc": low,
            "vah": high,
            "val": low,
            "value_area_percent": 70,
            "profile": [{"price": low, "volume": sum(b["volume"] for b in cleaned)}],
        }

    bin_count = min(max(int(bins), 12), 100)
    step = (high - low) / bin_count
    volumes = [0.0] * bin_count

    # Daily OHLCV does not contain intraday volume-at-price distribution.
    # Use the typical price (H+L+C)/3 as a deterministic approximation.
    for bar in cleaned:
        typical = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        index = int((typical - low) / step)
        index = min(max(index, 0), bin_count - 1)
        volumes[index] += bar["volume"]

    total_volume = sum(volumes)
    poc_index = max(range(bin_count), key=volumes)
    target = total_volume * 0.70
    included = {poc_index}
    accumulated = volumes[poc_index]

    # Expand the value area from POC by taking the larger adjacent volume.
    left = poc_index - 1
    right = poc_index + 1
    while accumulated < target and (left >= 0 or right < bin_count):
        left_volume = volumes[left] if left >= 0 else -1.0
        right_volume = volumes[right] if right < bin_count else -1.0
        if right_volume > left_volume:
            included.add(right)
            accumulated += right_volume
            right += 1
        else:
            included.add(left)
            accumulated += left_volume
            left -= 1

    val_index = min(included)
    vah_index = max(included)
    price_at = lambda i: low + (i + 0.5) * step

    profile = [
        {"price": round(price_at(i), 4), "volume": round(v, 2)}
        for i, v in enumerate(volumes)
        if v > 0
    ]
    ranked = sorted(
        ((price_at(i), v) for i, v in enumerate(volumes) if v > 0),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "bars_used": len(cleaned),
        "range_low": round(low, 4),
        "range_high": round(high, 4),
        "poc": round(price_at(poc_index), 4),
        "vah": round(price_at(vah_index), 4),
        "val": round(price_at(val_index), 4),
        "value_area_percent": 70,
        "total_volume": round(total_volume, 2),
        "high_volume_nodes": [
            {"price": round(price, 4), "volume": round(volume, 2)}
            for price, volume in ranked[:8]
        ],
        "profile": profile,
    }


def register_fvp_tool(server: FastMCP, client: httpx.AsyncClient) -> None:
    """Register the read-only Fixed Volume Profile tool."""

    @server.tool(
        annotations={
            "title": "Get Fixed Volume Profile",
            "readOnlyHint": True,
            "openWorldHint": True,
        }
    )
    async def get_fixed_volume_profile(
        symbol: str,
        lookback: int = 60,
        bins: int = 48,
        feed: str | None = None,
    ) -> dict[str, Any]:
        """Calculate a fixed-range volume profile from daily stock bars.

        Args:
            symbol: Stock ticker, e.g. NVDA or AAPL.
            lookback: Number of recent daily bars in the fixed range. Default 60.
            bins: Price bins used for the profile. Default 48.
            feed: Alpaca stock feed. Defaults to "iex" for paper/free access.

        Returns POC, VAH, VAL, range, high-volume nodes and the profile.
        This is a daily OHLCV approximation, not an exact TradingView clone.
        The tool is read-only and never places orders.
        """
        symbol_clean = symbol.upper().strip()
        if not symbol_clean:
            return {"error": {"message": "symbol is required"}}
        if lookback < 10:
            return {"error": {"message": "lookback must be at least 10 daily bars"}}
        if bins < 12:
            return {"error": {"message": "bins must be at least 12"}}

        data = await _fetch_daily_bars(client, symbol_clean, lookback, feed)
        if "error" in data:
            return data

        bars = data.get("bars", [])
        result = _profile(bars, bins)
        if "error" in result:
            result["symbol"] = symbol_clean
            result["feed"] = data.get("feed_used", (feed or "iex").strip().lower())
            result["request_start"] = data.get("request_start")
            result["bars_received"] = len(bars) if isinstance(bars, list) else 0
            return result

        result["symbol"] = symbol_clean
        result["timeframe"] = "1Day"
        result["feed"] = data.get("feed_used", (feed or "iex").strip().lower())
        result["method"] = "Typical-price allocation from daily OHLCV; 70% value area"
        result["lookback_requested"] = lookback
        result["bins_requested"] = bins
        result["request_start"] = data.get("request_start")
        return result
