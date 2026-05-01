"""Parameter optimizer — grid search + hill climbing.

Runs strategy templates with varying parameters across multiple symbols
and timeframes, collecting all results for the validation pipeline.
"""
from __future__ import annotations

import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import PIPELINE, SYMBOLS_BY_NAME, TIMEFRAMES
from ..data.parser import load_bars, add_indicators
from ..engine.backtest import _simulate, compute_metrics
from ..strategies.templates import StrategyTemplate, STRATEGIES

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Result of a single backtest trial."""
    strategy: str
    symbol: str
    timeframe: str
    params: dict[str, Any]
    metrics: dict
    equity: np.ndarray | None = None
    trades: np.ndarray | None = None
    error: str = ""

    @property
    def sharpe(self) -> float:
        return self.metrics.get("sharpe", 0.0)

    @property
    def max_dd(self) -> float:
        return self.metrics.get("max_drawdown", 1.0)

    @property
    def n_trades(self) -> int:
        return self.metrics.get("n_trades", 0)

    @property
    def is_valid(self) -> bool:
        return (
            not self.error
            and self.metrics
            and self.n_trades >= 10
            and self.max_dd < 1.0
        )


def _run_trial(
    strategy_name: str,
    symbol_name: str,
    timeframe: str,
    params: dict[str, Any],
    start: str | None = None,
    end: str | None = None,
) -> TrialResult:
    """Run one backtest trial (process-safe, called from ProcessPoolExecutor)."""
    try:
        symbol = SYMBOLS_BY_NAME[symbol_name]
        template = STRATEGIES[strategy_name]

        # Load data
        bars = load_bars(symbol_name, timeframe, start=start, end=end)
        if bars.empty:
            return TrialResult(strategy_name, symbol_name, timeframe, params, {},
                               error="no data")

        # Add indicators
        bars = add_indicators(bars)

        # Generate signals
        all_params = {**template.params, **params}
        signals = template.generate(bars, **all_params)

        # Filter: only valid entries
        positions = np.zeros_like(signals)
        in_pos = 0
        for i in range(len(signals)):
            sig = signals[i]
            if sig != 0 and in_pos == 0:
                positions[i] = sig
                in_pos = sig
            elif sig == -in_pos and in_pos != 0:
                positions[i] = -in_pos
                in_pos = 0

        # Prepare arrays for numba
        opens = bars["open"].values
        highs = bars["high"].values
        lows = bars["low"].values
        closes = bars["close"].values
        spreads = bars.get("spread_mean", pd.Series(symbol.typical_spread * symbol.point, index=bars.index)).values
        atr = bars.get("atr", pd.Series(0.001, index=bars.index)).values
        thin = bars.get("thin", pd.Series(False, index=bars.index)).values

        equity, pnl, trades_arr, dd = _simulate(
            positions, highs, lows, opens, closes, spreads, atr,
            symbol.lot_size, symbol.pip_value, symbol.commission,
            symbol.swap_long, symbol.swap_short, thin,
        )

        metrics = compute_metrics(equity, trades_arr, bars)

        return TrialResult(
            strategy_name, symbol_name, timeframe, params, metrics,
            equity=equity, trades=trades_arr,
        )

    except Exception as exc:
        logger.exception("Trial failed: %s/%s/%s %s", strategy_name, symbol_name, timeframe, params)
        return TrialResult(strategy_name, symbol_name, timeframe, params, {},
                           error=str(exc))


def grid_search(
    strategy_name: str,
    symbol_names: list[str],
    timeframes: list[str],
    n_samples: int = 50,
    start: str | None = None,
    end: str | None = None,
) -> list[TrialResult]:
    """Run parameter grid search across symbols and timeframes.

    n_samples: max parameter combinations to test (randomly sampled).
    """
    template = STRATEGIES[strategy_name]
    param_space = template.get_param_space()

    # Generate param combinations
    keys = list(param_space.keys())
    values = [param_space[k] for k in keys]
    all_combos = list(itertools.product(*values))

    logger.info(
        "Strategy '%s': %d param combinations × %d symbols × %d TFs = %d potential trials",
        strategy_name, len(all_combos), len(symbol_names), len(timeframes),
        len(all_combos) * len(symbol_names) * len(timeframes),
    )

    # Sample if too many
    if len(all_combos) > n_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_combos), n_samples, replace=False)
        all_combos = [all_combos[i] for i in idx]

    # Build trial queue
    trials = []
    for combo in all_combos:
        params = dict(zip(keys, combo))
        for sym in symbol_names:
            for tf in timeframes:
                trials.append((strategy_name, sym, tf, params, start, end))

    results: list[TrialResult] = []
    max_workers = PIPELINE["max_parallel"]

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_trial, *t): t for t in trials}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result.is_valid:
                results.append(result)
            if (i + 1) % 50 == 0:
                logger.info("  Trial %d/%d — %d valid so far", i + 1, len(trials), len(results))

    # Rank by Sharpe
    results.sort(key=lambda r: r.sharpe, reverse=True)
    return results


def optimize(
    strategies: list[str] | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    n_samples: int = 50,
    start: str | None = None,
    end: str | None = None,
) -> list[TrialResult]:
    """Main optimization entry point: grid search all strategies."""
    if strategies is None:
        strategies = list(STRATEGIES.keys())
    if symbols is None:
        symbols = ["EURUSD"]
    if timeframes is None:
        timeframes = ["H1"]

    all_results: list[TrialResult] = []
    for strat in strategies:
        logger.info("Optimizing: %s", strat)
        results = grid_search(strat, symbols, timeframes, n_samples, start, end)
        all_results.extend(results)
        if results:
            top = results[0]
            logger.info("  Top: Sharpe=%.2f DD=%.1f%% Trades=%d", top.sharpe, top.max_dd * 100, top.n_trades)

    all_results.sort(key=lambda r: r.sharpe, reverse=True)
    return all_results
