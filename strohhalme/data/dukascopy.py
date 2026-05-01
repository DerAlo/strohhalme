"""Dukascopy tick data downloader.

Dukascopy provides free historical tick data at:
  https://www.dukascopy.com/datafeed/{symbol}/{year}/{month}/{day}/{hour}h_ticks.bi5

Format: .bi5 files = LZMA-compressed binary
  Each tick: [timestamp_ms (int32 big-endian), ask (float32 big-endian), bid (float32 big-endian),
               ask_volume (float32), bid_volume (float32)] — 20 bytes per tick

Resources:
  - Downloads are ~1-5 MB per hour per symbol (~40-120 MB/day)
    → ~1-3 GB per symbol per year
  - Store raw .bi5 files temporarily → parse → delete raw → keep Parquet
  - Never download more than 2 symbols concurrently
  - Rate-limit: 1 request per 3 seconds (Dukascopy servers are fragile)
"""
from __future__ import annotations

import asyncio
import logging
import lzma
import struct
import time
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from ..config import DATA_RAW, DATA_PROCESSED, SYMBOLS_BY_NAME

logger = logging.getLogger(__name__)

DUKASCOPY_BASE = "https://www.dukascopy.com/datafeed"
DOWNLOAD_DELAY = 3.0       # seconds between requests
MAX_CONCURRENT = 2         # max concurrent downloads
TICK_STRUCT = struct.Struct(">IffII")  # ms from epoch, ask, bid, ask_vol, bid_vol


class DukascopyDownloader:
    """Downloads and parses Dukascopy tick data."""

    def __init__(self, cache_dir: Path = DATA_RAW):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session: aiohttp.ClientSession | None = None
        self._last_request = 0.0
        self._sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _rate_limited_get(self, url: str) -> bytes:
        """GET with mandatory delay between requests."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < DOWNLOAD_DELAY:
            await asyncio.sleep(DOWNLOAD_DELAY - elapsed)

        async with self._sem:
            if self._session is None:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30),
                )
            async with self._session.get(url) as resp:
                if resp.status == 404:
                    return b""  # no data for this hour (weekend, holiday)
                resp.raise_for_status()
                data = await resp.read()
            self._last_request = time.monotonic()
            return data

    def _url(self, symbol: str, dt: datetime) -> str:
        """Build Dukascopy datafeed URL for a specific hour."""
        return (
            f"{DUKASCOPY_BASE}/{symbol}/{dt.year}/"
            f"{dt.month - 1:02d}/{dt.day:02d}/"
            f"{dt.hour:02d}h_ticks.bi5"
        )

    def _parse_bi5(self, raw: bytes) -> np.ndarray | None:
        """Parse LZMA-compressed .bi5 bytes → numpy structured array.

        Returns: ndarray with fields ts, ask, bid, ask_vol, bid_vol
          ts = milliseconds since epoch
        """
        if not raw:
            return None
        try:
            decompressed = lzma.decompress(raw)
        except lzma.LZMAError:
            return None

        n = len(decompressed) // TICK_STRUCT.size
        if n == 0:
            return None

        data = np.zeros(n, dtype=[
            ("ts", "i8"),
            ("ask", "f8"),
            ("bid", "f8"),
            ("ask_vol", "f8"),
            ("bid_vol", "f8"),
        ])

        for i in range(n):
            offset = i * TICK_STRUCT.size
            ts, ask, bid, avol, bvol = TICK_STRUCT.unpack_from(decompressed, offset)
            data[i] = (int(ts), float(ask), float(bid), float(avol), float(bvol))

        # Ensure strict monotonic timestamps (Dukascopy sometimes has out-of-order ticks)
        if len(data) > 1:
            data.sort(order="ts")
            # Remove duplicates
            mask = np.ones(len(data), dtype=bool)
            mask[1:] = data["ts"][1:] != data["ts"][:-1]
            data = data[mask]

        return data if len(data) > 0 else None

    async def download_hour(self, symbol: str, dt: datetime) -> np.ndarray | None:
        """Download and parse one hour of tick data."""
        url = self._url(symbol, dt)
        raw = await self._rate_limited_get(url)
        return self._parse_bi5(raw)

    async def download_day(self, symbol: str, year: int, month: int, day: int) -> np.ndarray | None:
        """Download all 24 hours of a single day, concatenated."""
        tasks = []
        for hour in range(24):
            dt = datetime(year, month, day, hour)
            tasks.append(self.download_hour(symbol, dt))

        results = await asyncio.gather(*tasks)
        valid = [r for r in results if r is not None and len(r) > 0]
        if not valid:
            return None
        return np.concatenate(valid)

    async def download_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        progress: bool = True,
    ) -> pd.DataFrame | None:
        """Download tick data for a date range → DataFrame.

        Returns DataFrame with columns: ts, ask, bid, ask_vol, bid_vol
        Index: datetime (UTC)
        """
        if symbol not in SYMBOLS_BY_NAME:
            raise ValueError(f"Unknown symbol: {symbol}")

        all_ticks: list[np.ndarray] = []
        current = start.replace(hour=0, minute=0, second=0, microsecond=0)

        # Build list of days to download
        days: list[datetime] = []
        while current <= end:
            days.append(current)
            current += timedelta(days=1)

        if progress:
            days_iter = tqdm_asyncio(days, desc=f"Downloading {symbol}")
        else:
            days_iter = days

        for day_start in days_iter:
            ticks = await self.download_day(
                symbol, day_start.year, day_start.month, day_start.day,
            )
            if ticks is not None:
                all_ticks.append(ticks)

        if not all_ticks:
            return None

        combined = np.concatenate(all_ticks)
        combined.sort(order="ts")

        df = pd.DataFrame({
            "ask": combined["ask"],
            "bid": combined["bid"],
            "ask_vol": combined["ask_vol"],
            "bid_vol": combined["bid_vol"],
        }, index=pd.to_datetime(combined["ts"], unit="ms", utc=True))

        df["mid"] = (df["ask"] + df["bid"]) / 2.0
        df["spread"] = df["ask"] - df["bid"]
        return df

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None
