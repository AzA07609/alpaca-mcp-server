"""Read-only Fixed Range Volume Profile approximation."""

from __future__ import annotations

import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastmcp import FastMCP


def _error(message: str, **extra: Any) -> dict[str, Any]:
    error = {"message": message}
    error.update(extra)
    return {"error": error}


async def _fetch_daily_bars(
    client: httpx.AsyncClient, symbol: str, lookback: int, feed: str | None
) -> dict[str, Any]:
    selected_feed = (feed or "iex").strip().lower()
    # Use a sufficiently wide calendar window, then request the MOST RECENT
    # `lookback` bars with descending sort. The previous implementation used
    # sort=asc + limit=lookback, which returned the oldest bars in the window.
    calendar_days = max(int(lookback) * 3, int(lookback) + 30)
    start = datetime.now(timezone.utc) - timedelta(days=calendar_days)
    start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "timeframe": "1Day",
        "start": start_iso,
        "limit": min(max(int(lookback), 10), 1000),
        "sort": "desc",
        "adjustment": "raw",
        "feed": selected_feed,
    }
    path = f"/v2/stocks/{symbol.upper().strip()}/bars"
    try:
        response = await client.get(path, params=params)
        if response.is_error:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            return _error(
                "Alpaca market data request failed",
                http_status=response.status_code,
                detail=detail,
                feed_used=selected_feed,
                request_path=path,
                request_start=start_iso,
                request_sort="desc",
            )
        try:
            data = response.json()
        except ValueError:
            return _error(
                "Alpaca returned a non-JSON response",
                http_status=response.status_code,
                feed_used=selected_feed,
                request_path=path,
                request_start=start_iso,
                request_sort="desc",
            )
        bars = data.get("bars", []) if isinstance(data, dict) else []
        if isinstance(bars, dict):
            bars = bars.get(symbol.upper().strip(), [])
        if not isinstance(bars, list):
            bars = []
        # Return chronological order to all downstream indicator/profile code.
        bars.reverse()
        return {
            "bars": bars,
            "feed_used": selected_feed,
            "request_path": path,
            "request_start": start_iso,
            "request_sort": "desc_then_reverse_to_asc",
            "raw_keys": list(data.keys()) if isinstance(data, dict) else [],
        }
    except Exception as exc:
        return _error(
            "Unexpected exception while fetching market data",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(limit=8),
            feed_used=selected_feed,
            request_path=path,
            request_start=start_iso,
            request_sort="desc",
        )


def _price_at(index: int, low: float, step: float) -> float:
    return low + (index + 0.5) * step


def _profile(bars: list[dict[str, Any]], bins: int) -> dict[str, Any]:
    try:
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
                cleaned.append(
                    {"high": high, "low": low, "close": close, "volume": volume}
                )

        if not cleaned:
            return _error(
                "No usable OHLCV bars were returned.",
                bars_received=len(bars),
                bars_valid=0,
                stage="clean_ohlcv",
            )

        range_low = min(item["low"] for item in cleaned)
        range_high = max(item["high"] for item in cleaned)
        if range_high == range_low:
            total = sum(item["volume"] for item in cleaned)
            return {
                "bars_used": len(cleaned),
                "range_low": range_low,
                "range_high": range_high,
                "poc": range_low,
                "vah": range_high,
                "val": range_low,
                "value_area_percent": 70,
                "total_volume": total,
                "profile": [{"price": range_low, "volume": total}],
            }

        bin_count = min(max(int(bins), 12), 100)
        step = (range_high - range_low) / float(bin_count)
        volume_bins = [0.0 for _ in range(bin_count)]

        for item in cleaned:
            typical_price = (item["high"] + item["low"] + item["close"]) / 3.0
            index = int((typical_price - range_low) / step)
            if index < 0:
                index = 0
            elif index >= bin_count:
                index = bin_count - 1
            volume_bins[index] += item["volume"]

        total_volume = sum(volume_bins)
        poc_index = 0
        for index in range(1, bin_count):
            if volume_bins[index] > volume_bins[poc_index]:
                poc_index = index

        target_volume = total_volume * 0.70
        included = {poc_index}
        accumulated = volume_bins[poc_index]
        left = poc_index - 1
        right = poc_index + 1

        while accumulated < target_volume and (left >= 0 or right < bin_count):
            left_volume = volume_bins[left] if left >= 0 else -1.0
            right_volume = volume_bins[right] if right < bin_count else -1.0
            if right_volume > left_volume:
                included.add(right)
                accumulated += right_volume
                right += 1
            else:
                if left >= 0:
                    included.add(left)
                    accumulated += left_volume
                left -= 1

        val_index = min(included)
        vah_index = max(included)

        profile = []
        for index, volume in enumerate(volume_bins):
            if volume > 0:
                profile.append(
                    {
                        "price": round(_price_at(index, range_low, step), 4),
                        "volume": round(volume, 2),
                    }
                )

        ranked = []
        for index, volume in enumerate(volume_bins):
            if volume > 0:
                ranked.append((_price_at(index, range_low, step), volume))
        ranked.sort(key=lambda pair: pair[1], reverse=True)

        return {
            "bars_used": len(cleaned),
            "range_low": round(range_low, 4),
            "range_high": round(range_high, 4),
            "poc": round(_price_at(poc_index, range_low, step), 4),
            "vah": round(_price_at(vah_index, range_low, step), 4),
            "val": round(_price_at(val_index, range_low, step), 4),
            "value_area_percent": 70,
            "total_volume": round(total_volume, 2),
            "high_volume_nodes": [
                {"price": round(price, 4), "volume": round(volume, 2)}
                for price, volume in ranked[:8]
            ],
            "profile": profile,
        }
    except Exception as exc:
        return _error(
            "FVP calculation failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(limit=12),
            stage="profile_calculation",
            bars_received=len(bars),
        )


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
        """Calculate a fixed-range volume profile from the most recent daily stock bars."""
        try:
            symbol_clean = symbol.upper().strip()
            if not symbol_clean:
                return _error("symbol is required")
            if lookback < 10:
                return _error("lookback must be at least 10 daily bars")
            if bins < 12:
                return _error("bins must be at least 12")

            data = await _fetch_daily_bars(client, symbol_clean, lookback, feed)
            if "error" in data:
                return data

            bars = data.get("bars", [])
            result = _profile(bars, bins)
            if "error" in result:
                result["symbol"] = symbol_clean
                result["feed"] = data.get("feed_used")
                result["request_path"] = data.get("request_path")
                result["request_start"] = data.get("request_start")
                result["request_sort"] = data.get("request_sort")
                result["bars_received"] = len(bars) if isinstance(bars, list) else 0
                result["raw_keys"] = data.get("raw_keys", [])
                return result

            result["symbol"] = symbol_clean
            result["timeframe"] = "1Day"
            result["feed"] = data.get("feed_used")
            result["method"] = "Typical-price allocation from daily OHLCV; 70% value area"
            result["lookback_requested"] = lookback
            result["bins_requested"] = bins
            result["request_path"] = data.get("request_path")
            result["request_start"] = data.get("request_start")
            result["request_sort"] = data.get("request_sort")
            result["first_bar_date"] = bars[0].get("t") if bars else None
            result["last_bar_date"] = bars[-1].get("t") if bars else None
            return result
        except Exception as exc:
            return _error(
                "get_fixed_volume_profile handler failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(limit=12),
                stage="handler",
            )
