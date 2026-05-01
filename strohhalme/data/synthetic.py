"""Synthetic data generator for testing when Dukascopy is unavailable.

Generates realistic-looking OHLCV bars for any symbol/timeframe.
Used as fallback when data download fails (e.g., cloud IPs blocked).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from ..config import DATA_PROCESSED, SYMBOLS_BY_NAME, TIMEFRAMES


def generate_synthetic(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    seed: int = 42,
    volatility_pct: float = 12.0,
    base_dir: Path = DATA_PROCESSED,
) -> tuple[pd.DataFrame, Path]:
    """Generate synthetic OHLCV data and save as Parquet.

    Returns (bars_df, output_path).
    """
    rng = np.random.default_rng(seed)
    sym = SYMBOLS_BY_NAME[symbol]
    tf_min = TIMEFRAMES[timeframe]

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    n_bars = int((end_dt - start_dt).total_seconds() / (tf_min * 60))

    dates = pd.date_range(start_dt, periods=n_bars, freq=f'{tf_min}min')
    if len(dates) > n_bars:
        dates = dates[:n_bars]

    # Geometric Brownian motion with mean reversion and trend regimes
    initial_price = 1.10 if "EUR" in symbol else 150.0 if "JPY" in symbol else 1.30
    sigma = (volatility_pct / 100) / np.sqrt(252 * 24 * 60 / tf_min)

    # Generate realistic market regimes: trend + mean-reversion cycles
    returns = np.zeros(n_bars)
    regime_length = n_bars // 8  # ~8 regimes over the period
    for regime_start in range(0, n_bars, regime_length):
        end = min(regime_start + regime_length, n_bars)
        seg_len = end - regime_start
        # Alternate: trending, mean-reverting, volatile, calm
        regime_type = (regime_start // regime_length) % 4
        if regime_type == 0:
            drift = sigma * 0.3  # mild uptrend
        elif regime_type == 1:
            drift = -sigma * 0.2  # mild downtrend
        elif regime_type == 2:
            drift = 0.0  # mean-reverting / ranging
        else:
            drift = 0.0  # calm / low vol
        returns[regime_start:end] = rng.normal(drift, sigma * (1.5 if regime_type == 2 else 1.0), seg_len)

    # Add technical patterns: occasional EMA crossovers, breakouts, etc.
    close = initial_price * np.exp(np.cumsum(returns))

    # Inject sine-wave component for detectable technical patterns
    pattern_amp = initial_price * 0.02
    pattern_freq = rng.uniform(0.001, 0.005)
    close = close + pattern_amp * np.sin(pattern_freq * np.arange(n_bars))

    # OHLC envelope
    noise_scale = sigma * 0.5
    high = close + np.abs(rng.normal(0, noise_scale, n_bars))
    low = close - np.abs(rng.normal(0, noise_scale, n_bars))
    open_p = close - rng.normal(0, noise_scale * 0.3, n_bars)

    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(n_bars):
        hi = max(open_p[i], close[i])
        lo = min(open_p[i], close[i])
        high[i] = max(high[i], hi + 1e-5)
        low[i] = min(low[i], lo - 1e-5)

    df = pd.DataFrame({
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'tick_count': rng.integers(20, 200, n_bars),
        'volume': rng.exponential(100, n_bars),
        'spread_close': np.full(n_bars, sym.typical_spread * sym.point),
        'spread_mean': np.full(n_bars, sym.typical_spread * sym.point),
        'thin': np.zeros(n_bars, dtype=bool),
    }, index=dates)

    out_dir = base_dir / symbol / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)
    year = start_dt.year
    path = out_dir / f'{symbol}_{timeframe}_{year}.parquet'
    df.to_parquet(path, compression='snappy')

    return df, path


def ensure_data(
    symbol: str,
    timeframe: str,
    start: str = '2024-01-01',
    end: str = '2024-06-30',
) -> pd.DataFrame:
    """Ensure data exists, generating synthetic if not found."""
    from .parser import load_bars

    try:
        bars = load_bars(symbol, timeframe, start=start, end=end)
        if not bars.empty:
            return bars
    except FileNotFoundError:
        pass

    print(f"Generating synthetic {symbol} {timeframe} data...")
    bars, _ = generate_synthetic(symbol, timeframe, start, end)
    return bars
