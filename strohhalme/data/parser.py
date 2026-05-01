"""Tick → OHLCV bar aggregator + Parquet storage.

Converts Dukascopy tick DataFrames to OHLCV bars at configured timeframes,
stores as compressed Parquet files partitioned by symbol/year.

Parquet with snappy compression achieves ~6:1 compression ratio on OHLCV data.
Typical: 1 year EURUSD M15 = ~35,000 bars → ~700KB stored.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from ..config import DATA_PROCESSED, TIMEFRAMES, SYMBOLS_BY_NAME

logger = logging.getLogger(__name__)


def ticks_to_bars(
    ticks: pd.DataFrame,
    timeframe: str,
    symbol: str | None = None,
    min_ticks: int = 5,
) -> pd.DataFrame:
    """Aggregate tick data → OHLCV bars.

    Args:
        ticks: DataFrame with datetime index, columns: ask, bid, mid, spread
        timeframe: e.g. "M15", "H1"
        symbol: for pip point calculations (optional)
        min_ticks: minimum ticks per bar (fewer = low-liquidity, mark bar)

    Returns:
        DataFrame with OHLCV columns, datetime index
    """
    tf_minutes = TIMEFRAMES[timeframe]

    # Use mid price for OHLC (avoid spread bouncing)
    ohlc = ticks["mid"].resample(f"{tf_minutes}min", label="right", closed="right")

    df = pd.DataFrame({
        "open": ohlc.first(),
        "high": ohlc.max(),
        "low": ohlc.min(),
        "close": ohlc.last(),
        "tick_count": ohlc.count(),
    })

    # Volume: sum of tick-weighted volume (use mid of ask/bid vol)
    if "ask_vol" in ticks.columns and "bid_vol" in ticks.columns:
        ticks["volume"] = (ticks["ask_vol"] + ticks["bid_vol"]) / 2
        df["volume"] = ticks["volume"].resample(f"{tf_minutes}min").sum()
    else:
        df["volume"] = df["tick_count"]  # fallback

    # Spread at bar close
    df["spread_close"] = ticks["spread"].resample(f"{tf_minutes}min").last()
    df["spread_mean"] = ticks["spread"].resample(f"{tf_minutes}min").mean()

    # Flag low-liquidity bars
    df["thin"] = df["tick_count"] < min_ticks

    # Drop bars with no ticks (weekends, holidays)
    df = df.dropna(subset=["open", "close"])

    return df


def bars_to_parquet(
    bars: pd.DataFrame,
    symbol: str,
    timeframe: str,
    base_dir: Path = DATA_PROCESSED,
) -> Path:
    """Store OHLCV bars as Parquet, partitioned by symbol/year.

    File: {base_dir}/{symbol}/{timeframe}/{symbol}_{timeframe}_{year}.parquet
    """
    if bars.empty:
        logger.warning(f"No bars to store for {symbol} {timeframe}")
        return Path()

    out_dir = base_dir / symbol / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write one file per year for efficient filtering
    for year, group in bars.groupby(bars.index.year):
        if group.empty:
            continue
        path = out_dir / f"{symbol}_{timeframe}_{year}.parquet"
        group.to_parquet(path, compression="snappy", index=True)
        logger.debug(f"Wrote {len(group):,} bars → {path}")

    return out_dir


def load_bars(
    symbol: str,
    timeframe: str,
    start: str | None = None,
    end: str | None = None,
    base_dir: Path = DATA_PROCESSED,
) -> pd.DataFrame:
    """Load OHLCV bars from Parquet store.

    Args:
        symbol: e.g. "EURUSD"
        timeframe: e.g. "M15"
        start, end: ISO date strings, e.g. "2020-01-01"
    """
    data_dir = base_dir / symbol / timeframe
    if not data_dir.exists():
        raise FileNotFoundError(f"No data for {symbol}/{timeframe} at {data_dir}")

    files = sorted(data_dir.glob(f"{symbol}_{timeframe}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs).sort_index()
    # Remove duplicates (parquet file boundaries may overlap)
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def add_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators (for strategy templates).

    Operates on a copy, does not mutate input.
    All indicators are backward-looking (no lookahead bias).
    """
    df = bars.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ATR (14-period)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=7).mean()

    # SMA
    for p in [20, 50, 200]:
        df[f"sma_{p}"] = close.rolling(p, min_periods=p // 2).mean()

    # EMA
    for p in [9, 21, 55]:
        df[f"ema_{p}"] = close.ewm(span=p, min_periods=p // 2).mean()

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # Bollinger Bands (20,2)
    df["bb_mid"] = df["sma_20"]
    bb_std = close.rolling(20, min_periods=10).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # MACD (12,26,9)
    ema12 = close.ewm(span=12, min_periods=6).mean()
    ema26 = close.ewm(span=26, min_periods=13).mean()
    macd_line = ema12 - ema26
    df["macd"] = macd_line
    df["macd_signal"] = macd_line.ewm(span=9, min_periods=5).mean()
    df["macd_hist"] = macd_line - df["macd_signal"]

    return df
