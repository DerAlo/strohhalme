"""Strategy templates and signal generators.

Each strategy is a function that takes a DataFrame of bars (with indicators)
and returns a numpy array of position signals:
  -1 = go short, 0 = flat, 1 = go long

Signals are filtered: same-direction signals while already in position are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd


# ── Signal generation primitives ─────────────────────────────────────────────

def crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return 1 where a crosses above b, -1 where a crosses below b, 0 otherwise."""
    above = a > b
    above_prev = np.roll(above, 1)
    cross_up = above & ~above_prev
    cross_down = ~above & above_prev
    cross_up[0] = False
    cross_down[0] = False
    return cross_up.astype(np.int8) - cross_down.astype(np.int8)


def above(a: np.ndarray, b: np.ndarray, lag: int = 0) -> np.ndarray:
    """Return True where a > b, with optional lag."""
    if lag == 0:
        return a > b
    return a > np.roll(b, lag)


def derivative(series: np.ndarray, period: int = 1) -> np.ndarray:
    """First difference."""
    return series - np.roll(series, period)


def rolling_max(series: np.ndarray, window: int) -> np.ndarray:
    """Rolling max (backward-looking, no lookahead)."""
    result = np.full_like(series, np.nan)
    for i in range(window - 1, len(series)):
        result[i] = np.max(series[i - window + 1:i + 1])
    return result


def rolling_min(series: np.ndarray, window: int) -> np.ndarray:
    """Rolling min (backward-looking)."""
    result = np.full_like(series, np.nan)
    for i in range(window - 1, len(series)):
        result[i] = np.min(series[i - window + 1:i + 1])
    return result


# ── Strategy Template Registry ───────────────────────────────────────────────

@dataclass
class StrategyTemplate:
    """A parameterized strategy template."""
    name: str
    description: str
    params: dict[str, Any]       # default parameter values
    param_ranges: dict[str, tuple[float, float]]  # (min, max) for optimization
    generate: Callable[..., np.ndarray]  # fn(bars, **params) → signals

    def get_param_space(self) -> dict[str, np.ndarray]:
        """Generate parameter combinations for grid search."""
        space = {}
        for name, (lo, hi) in self.param_ranges.items():
            default = self.params.get(name, (lo + hi) / 2)
            if isinstance(default, float):
                # Float param: 5 steps log-spaced or linear
                space[name] = np.linspace(lo, hi, 5)
            elif isinstance(default, int):
                step = max(1, int((hi - lo) / 10))
                space[name] = np.arange(int(lo), int(hi) + 1, step)
            elif isinstance(default, bool):
                space[name] = np.array([False, True])
        return space


# ── Built-in Strategies ──────────────────────────────────────────────────────

def _ema_crossover(bars: pd.DataFrame, fast: int = 9, slow: int = 21, filter_rsi: float = 0.0) -> np.ndarray:
    """Classic EMA crossover."""
    signals = crossover(bars[f"ema_{fast}"].values, bars[f"ema_{slow}"].values)
    if filter_rsi > 0:
        rsi = bars["rsi"].values
        signals[(signals == 1) & (rsi > 70)] = 0
        signals[(signals == -1) & (rsi < 30)] = 0
    return signals


def _macd_signal(bars: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD histogram crossover with signal line."""
    hist = bars["macd_hist"].values
    return np.where(hist > 0, 1, -1)


def _bb_breakout(bars: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
                  atr_filter: float = 0.0) -> np.ndarray:
    """Bollinger Band mean-reversion: buy below lower, sell above upper."""
    signals = np.zeros(len(bars), dtype=np.int8)
    close = bars["close"].values
    upper = bars["bb_upper"].values
    lower = bars["bb_lower"].values

    # Only enter when price is outside bands
    signals[close < lower] = 1
    signals[close > upper] = -1

    # Exit on return to mid (SMA)
    mid = bars["bb_mid"].values
    in_position = np.zeros(len(bars), dtype=np.int8)
    prev_pos = 0
    for i in range(len(bars)):
        if signals[i] != 0:
            prev_pos = signals[i]
        elif prev_pos != 0:
            if (prev_pos == 1 and close[i] >= mid[i]) or (prev_pos == -1 and close[i] <= mid[i]):
                prev_pos = 0
                signals[i] = -signals[i] if signals[i] == 0 else signals[i]  # exit signal
        in_position[i] = prev_pos

    return signals


def _trend_follower(bars: pd.DataFrame, sma_fast: int = 50, sma_slow: int = 200,
                    atr_mult: float = 1.5) -> np.ndarray:
    """Trend follower with ATR trailing stop."""
    close = bars["close"].values
    sma_f = bars[f"sma_{sma_fast}"].values
    sma_s = bars[f"sma_{sma_slow}"].values
    atr = bars["atr"].values

    signals = np.zeros(len(bars), dtype=np.int8)
    in_position = 0
    stop_level = 0.0

    for i in range(max(sma_slow, 200), len(bars)):
        trend_up = sma_f[i] > sma_s[i]
        trend_down = sma_f[i] < sma_s[i]

        if in_position == 0:
            if trend_up and close[i] > sma_f[i]:
                in_position = 1
                stop_level = close[i] - atr_mult * atr[i]
                signals[i] = 1
            elif trend_down and close[i] < sma_f[i]:
                in_position = -1
                stop_level = close[i] + atr_mult * atr[i]
                signals[i] = -1
        elif in_position == 1:
            if close[i] <= stop_level or trend_down:
                in_position = 0
                signals[i] = -1
            else:
                stop_level = max(stop_level, close[i] - atr_mult * atr[i])
        elif in_position == -1:
            if close[i] >= stop_level or trend_up:
                in_position = 0
                signals[i] = 1
            else:
                stop_level = min(stop_level, close[i] + atr_mult * atr[i])

    return signals


def _asian_session_fade(bars: pd.DataFrame, atr_mult: float = 0.5) -> np.ndarray:
    """Fade the Asian session breakout at London open (mean-reversion)."""
    signals = np.zeros(len(bars), dtype=np.int8)

    # This operates on H1 bars — detects 22:00-07:00 UTC range,
    # then fades any breakout > ATR from the range.
    atr = bars["atr"].values
    high = bars["high"].values
    low = bars["low"].values
    close = bars["close"].values

    for i in range(24, len(bars)):
        hour = bars.index[i].hour
        if hour == 7:  # 07:00 UTC = London open (08:00 BST)
            asian_high = np.max(high[i-8:i])   # 23:00-07:00
            asian_low = np.min(low[i-8:i])
            asian_range = asian_high - asian_low

            if close[i] > asian_high + atr_mult * atr[i]:
                signals[i] = -1  # fade upside breakout
            elif close[i] < asian_low - atr_mult * atr[i]:
                signals[i] = 1   # fade downside breakout

    return signals


# ── Registry ─────────────────────────────────────────────────────────────────

STRATEGIES: dict[str, StrategyTemplate] = {
    "ema_crossover": StrategyTemplate(
        name="ema_crossover",
        description="EMA crossover with optional RSI filter",
        params={"fast": 9, "slow": 21, "filter_rsi": 0.0},
        param_ranges={"fast": (5, 30), "slow": (15, 75), "filter_rsi": (0.0, 0.5)},
        generate=_ema_crossover,
    ),
    "macd_signal": StrategyTemplate(
        name="macd_signal",
        description="MACD histogram direction trading",
        params={"fast": 12, "slow": 26, "signal": 9},
        param_ranges={"fast": (8, 20), "slow": (20, 40), "signal": (5, 15)},
        generate=_macd_signal,
    ),
    "bb_breakout": StrategyTemplate(
        name="bb_breakout",
        description="Bollinger Band mean-reversion",
        params={"bb_period": 20, "bb_std": 2.0, "atr_filter": 0.0},
        param_ranges={"bb_period": (10, 50), "bb_std": (1.5, 3.0), "atr_filter": (0.0, 2.0)},
        generate=_bb_breakout,
    ),
    "trend_follower": StrategyTemplate(
        name="trend_follower",
        description="SMA trend follower with ATR trailing stop",
        params={"sma_fast": 50, "sma_slow": 200, "atr_mult": 1.5},
        param_ranges={"sma_fast": (20, 100), "sma_slow": (100, 400), "atr_mult": (0.5, 4.0)},
        generate=_trend_follower,
    ),
    "asian_fade": StrategyTemplate(
        name="asian_fade",
        description="Mean-reversion fade of Asian session breakout",
        params={"atr_mult": 0.5},
        param_ranges={"atr_mult": (0.25, 2.0)},
        generate=_asian_session_fade,
    ),
}
