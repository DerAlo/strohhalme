"""Vectorized backtesting engine.

Design goals:
  - Lightweight: no dependency on MT5, purely Python + numpy/numba
  - Vectorized: processes all bars in one pass (no per-tick Python loop)
  - Realistic costs: spread, slippage, commission, swap, thin-market detection
  - Multi-position: can handle both long/flat and long/short/flat modes

Architecture:
  Strategies return arrays of signals (1=buy, -1=sell, 0=flat).
  The engine simulates fills, manages equity curve, and computes metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from numba import njit

from ..config import SYMBOLS_BY_NAME

logger = logging.getLogger(__name__)


# ── Numba-accelerated core loop ─────────────────────────────────────────────
# The fill simulation must be iterative (you can't buy twice on the same signal
# without tracking position state). Numba makes this ~100x faster than Python.

@njit(cache=True)
def _simulate(
    positions: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    spreads: np.ndarray,
    atr: np.ndarray,
    lot_size: float,
    pip_value: float,
    commission: float,
    swap_long: float,
    swap_short: float,
    thin_bars: np.ndarray,
    slippage_pct: float = 0.3,
    min_slippage: float = 0.5,
    max_slippage: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-jitted fill simulation loop.

    Returns: (equity, pnl_per_bar, trades, drawdown)
    All returns are np.float64 arrays sized len(positions).
    """
    n = len(positions)
    equity = np.zeros(n, dtype=np.float64)
    pnl_bar = np.zeros(n, dtype=np.float64)
    trades = np.zeros(n, dtype=np.float64)
    drawdown = np.zeros(n, dtype=np.float64)

    in_position = False
    position_type = 0         # 1=long, -1=short
    entry_price = 0.0
    entry_bar = 0
    peak_equity = 0.0
    current_equity = 0.0
    swap_counter = 0

    for i in range(n):
        signal = positions[i]
        spread = spreads[i]
        is_thin = thin_bars[i]
        slippage = max(min_slippage, min(max_slippage, atr[i] * slippage_pct))

        if signal != 0 and not in_position:
            # ── ENTRY ──
            if signal == 1:  # buy
                fill_price = opens[i] + spread  # always buy at ask
                position_type = 1
            else:  # sell
                fill_price = opens[i]  # sell at bid (open - spread bot for short)
                position_type = -1

            in_position = True
            entry_price = fill_price
            entry_bar = i
            swap_counter = 0

            # Commission on entry
            current_equity -= commission
            pnl_bar[i] -= commission

        elif signal == -position_type and in_position:
            # ── EXIT (reverse signal) ──
            if position_type == 1:  # close long
                fill_price = opens[i] - slippage  # sell at bid minus slippage
            else:  # close short
                fill_price = opens[i] + spread + slippage  # buy back at ask plus slippage

            # P&L
            pnl = (fill_price - entry_price) * position_type * lot_size
            current_equity += pnl - commission
            pnl_bar[i] = pnl - commission
            trades[i] = pnl - commission

            in_position = False
            position_type = 0

        elif in_position:
            # ── HOLD ──
            # Mark to market at bar close (mid price, no spread)
            mtm = (closes[i] - entry_price) * position_type * lot_size
            pnl_bar[i] = mtm

            # Swap: charge every 24 bars ≈ daily on H1
            swap_counter += 1
            if swap_counter >= 24:
                swap_rate = swap_long if position_type == 1 else swap_short
                current_equity += swap_rate
                pnl_bar[i] += swap_rate
                swap_counter = 0

        # Update equity
        current_equity += pnl_bar[i] - (pnl_bar[i-1] if i > 0 else 0.0)
        equity[i] = current_equity

        # Drawdown
        if current_equity > peak_equity:
            peak_equity = current_equity
        dd = peak_equity - current_equity
        drawdown[i] = dd

    return equity, pnl_bar, trades, drawdown


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    equity: np.ndarray,
    trades: np.ndarray,
    bars: pd.DataFrame,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> dict:
    """Compute standard performance metrics from equity curve."""
    if len(equity) < 20:
        return {"error": "too few bars"}

    n = len(equity)
    years = n / (trading_days * 24 * 60 / TIMEFRAME_TO_MINUTES.get(bars.index.freqstr, 60))

    # Returns (log, per bar)
    rets = np.diff(np.log(np.maximum(equity + 1e-10, 1e-10)))
    rets = rets[np.isfinite(rets)]

    if len(rets) < 5:
        return {"error": "too few returns"}

    # Basic
    total_return = (equity[-1] / equity[0] - 1) if equity[0] > 0 else 0.0
    cagr = (equity[-1] / equity[0]) ** (1 / max(years, 0.5)) - 1.0 if equity[0] > 0 else 0.0

    # Risk
    annual_vol = np.std(rets) * np.sqrt(trading_days * 24 * 60)
    downside_rets = rets[rets < 0]
    sortino_vol = np.std(downside_rets) * np.sqrt(trading_days * 24 * 60) if len(downside_rets) > 0 else annual_vol
    sharpe = (cagr - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0
    sortino = (cagr - risk_free_rate) / sortino_vol if sortino_vol > 0 else 0.0

    # Drawdown
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(np.max(dd))
    dd_duration = 0
    max_dd_duration = 0
    for v in dd:
        if v > 0:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    # Trade stats
    trade_mask = trades != 0
    trade_values = trades[trade_mask]
    n_trades = len(trade_values)
    win_rate = float(np.mean(trade_values > 0)) if n_trades > 0 else 0.0
    avg_win = float(np.mean(trade_values[trade_values > 0])) if np.any(trade_values > 0) else 0.0
    avg_loss = float(np.mean(trade_values[trade_values < 0])) if np.any(trade_values < 0) else 0.0
    profit_factor = abs(avg_win * win_rate / (avg_loss * (1 - win_rate))) if avg_loss != 0 and win_rate > 0 else 0.0

    # Calmar ratio
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    return {
        "cagr": round(cagr, 4),
        "total_return": round(total_return, 4),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "max_drawdown": round(max_dd, 4),
        "max_dd_bars": max_dd_duration,
        "annual_vol": round(annual_vol, 4),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "years": round(years, 2),
    }


TIMEFRAME_TO_MINUTES = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "1h": 60, "4h": 240, "1D": 1440}
