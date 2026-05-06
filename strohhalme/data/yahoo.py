"""Yahoo Finance OHLC data downloader.

Fallback data source when Dukascopy is unreachable.
Yahoo Finance provides free OHLC data for forex pairs.

Limitations:
  - H1 data: available for last ~730 days
  - M15 data: available for last ~60 days
  - Daily data: full history available

Usage:
    from .yahoo import YahooDownloader
    dl = YahooDownloader()
    bars = await dl.download_bars("EURUSD", "2024-01-01", "2024-12-31", "H1")
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Map internal timeframe codes to yfinance interval strings
INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "M15": "15m",
    "30m": "30m",
    "H1": "1h",
    "D1": "1d",
    "H2": "2h",
    "H4": "4h",
    "D": "1d",
    "W": "1wk",
    "M": "1mo",
}

# Yahoo Finance max history windows per interval
MAX_DAYS = {
    "1m": 7,
    "5m": 30,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "2h": 730,
    "4h": 730,
    "1d": 10 * 365,
}

YAHOO_TICKER = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "ETHUSD": "ETH-USD",
    "BTCUSD": "BTC-USD",
}


class YahooDownloader:
    """Downloads OHLC bar data from Yahoo Finance."""

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}

    async def download_bars(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        timeframe: str,
    ) -> pd.DataFrame | None:
        """Download OHLC bars for a symbol over a date range.

        Downloads in batched chunks if the range exceeds Yahoo's window.
        Returns DataFrame with columns: open, high, low, close, volume
        Index: datetime (UTC)
        """
        ticker = YAHOO_TICKER.get(symbol)
        if ticker is None:
            raise ValueError(f"Unsupported symbol: {symbol} (supported: {list(YAHOO_TICKER.keys())})")

        interval = INTERVAL_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        if isinstance(start, str):
            start = pd.Timestamp(start).to_pydatetime()
        if isinstance(end, str):
            end = pd.Timestamp(end).to_pydatetime()

        max_days = MAX_DAYS.get(interval, 30)
        all_bars: list[pd.DataFrame] = []

        # Download in batches
        batch_start = start
        while batch_start < end:
            batch_end = min(batch_start + timedelta(days=max_days), end)
            logger.info(
                "Yahoo: downloading %s %s %s → %s",
                symbol, timeframe, batch_start.date(), batch_end.date(),
            )

            df = yf.download(
                ticker,
                start=batch_start,
                end=batch_end + timedelta(days=1),
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if df is not None and not df.empty:
                # Flatten MultiIndex columns if present (yfinance returns MultiIndex)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].lower() for col in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                keep = ["open", "high", "low", "close", "volume"]
                df = df[[c for c in keep if c in df.columns]]
                all_bars.append(df)

            batch_start = batch_end

        if not all_bars:
            logger.warning("Yahoo: no data returned for %s %s", symbol, timeframe)
            return None

        result = pd.concat(all_bars).sort_index()
        # Remove duplicates (from overlap between batches)
        result = result[~result.index.duplicated(keep="first")]

        result.index = result.index.tz_convert("UTC")
        result.index.name = "time"
        return result
