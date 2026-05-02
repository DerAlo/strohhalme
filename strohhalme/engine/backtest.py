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
    lot_size: float,           # base lot size (100000 = standard)
    pip_value: float,          # $ per pip per standard lot
    commission: float,         # per round-turn per STANDARD lot
    swap_long: float,
    swap_short: float,
    thin_bars: np.ndarray,
    slippage_pct: float = 0.3,
    min_slippage: float = 0.5,
    max_slippage: float = 50.0,
    initial_equity: float = 10000.0,
    risk_per_trade: float = 0.01,
    stop_atr: float = 2.0,
    min_lot: float = 1000.0,
    max_lot: float = 100000.0,
    stop_out_pct: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-jitted fill simulation loop with risk-based position sizing.

    Instead of a fixed lot size, each trade's position is sized dynamically
    so that at most risk_per_trade of current equity is lost if price moves
    stop_atr × ATR against the position.  This prevents account blowup from
    a single bad trade and keeps the simulation realistic.

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
    peak_equity = initial_equity
    current_equity = initial_equity
    swap_counter = 0
    stopped_out = False
    stop_out_threshold = initial_equity * stop_out_pct

    active_lot = min_lot  # last computed lot size (in units, e.g. 1000 = micro lot)

    for i in range(n):
        if stopped_out:
            equity[i] = current_equity
            pnl_bar[i] = 0.0
            continue

        signal = positions[i]
        spread = spreads[i]
        is_thin = thin_bars[i]
        slippage = max(min_slippage, min(max_slippage, atr[i] * slippage_pct))

        if signal != 0 and not in_position:
            # ── ENTRY ── Compute risk-based position size
            # risk = stop_distance (abs price) × position_units
            # So: position_units = risk_amount / stop_distance
            risk_amount = current_equity * risk_per_trade
            stop_dist = atr[i] * stop_atr
            if stop_dist > 0.0 and risk_amount > 0.0:
                active_lot = max(min_lot, min(max_lot, risk_amount / stop_dist))
            else:
                active_lot = min_lot

            if signal == 1:  # buy
                fill_price = opens[i] + spread
                position_type = 1
            else:  # sell
                fill_price = opens[i]
                position_type = -1

            in_position = True
            entry_price = fill_price
            entry_bar = i
            swap_counter = 0

            # Commission on entry (proportional to active lot)
            entry_comm = commission * (active_lot / lot_size)
            current_equity -= entry_comm
            pnl_bar[i] -= entry_comm

        elif signal == -position_type and in_position:
            # ── EXIT (reverse signal) ──
            if position_type == 1:  # close long
                fill_price = opens[i] - slippage
            else:  # close short
                fill_price = opens[i] + spread + slippage

            pnl = (fill_price - entry_price) * position_type * active_lot
            exit_comm = commission * (active_lot / lot_size)
            current_equity += pnl - exit_comm
            pnl_bar[i] = pnl - exit_comm
            trades[i] = pnl - exit_comm

            in_position = False
            position_type = 0

        elif in_position:
            # ── HOLD ── Mark to market
            mtm = (closes[i] - entry_price) * position_type * active_lot
            prev_mtm = (closes[i-1] - entry_price) * position_type * active_lot if i > 0 else 0.0
            pnl_bar[i] = mtm - prev_mtm

            swap_counter += 1
            if swap_counter >= 24:
                swap_rate = swap_long if position_type == 1 else swap_short
                swap_rate_scaled = swap_rate * (active_lot / lot_size)
                current_equity += swap_rate_scaled
                pnl_bar[i] += swap_rate_scaled
                swap_counter = 0

        # Update equity
        current_equity += pnl_bar[i]
        equity[i] = current_equity

        # Circuit breaker: force-close at stop_out
        if current_equity < stop_out_threshold and in_position:
            if position_type == 1:
                close_price = closes[i] - slippage
            else:
                close_price = closes[i] + spread + slippage
            pnl = (close_price - entry_price) * position_type * active_lot
            exit_comm = commission * (active_lot / lot_size)
            current_equity += pnl - exit_comm
            pnl_bar[i] += pnl - exit_comm
            trades[i] = pnl - exit_comm
            in_position = False
            position_type = 0
            equity[i] = current_equity
        if current_equity < stop_out_threshold and not in_position:
            stopped_out = True

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

    # Basic — safe against negative equity (blowup) or zero equity
    if equity[0] > 0 and equity[-1] > 0:
        total_return = equity[-1] / equity[0] - 1
        cagr = (equity[-1] / equity[0]) ** (1 / max(years, 0.5)) - 1.0
    else:
        total_return = -1.0
        cagr = -1.0

    # Risk — annualize based on actual bar frequency
    minutes_per_bar = TIMEFRAME_TO_MINUTES.get(bars.index.freqstr, 60) if hasattr(bars.index, 'freqstr') else 60
    bars_per_year = trading_days * 24 * 60 / minutes_per_bar
    annual_vol = np.std(rets) * np.sqrt(bars_per_year)
    downside_rets = rets[rets < 0]
    sortino_vol = np.std(downside_rets) * np.sqrt(bars_per_year) if len(downside_rets) > 0 else annual_vol
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


TIMEFRAME_TO_MINUTES = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "1h": 60, "4h": 240, "D1": 1440, "1D": 1440}
