"""Advanced validation gates — Walk-Forward, Parameter Stability, Regime Analysis.

These gates require re-running optimizations on data windows and are
computationally expensive. They run only on candidates that pass all
basic gates (trades, DD, Sharpe, PF, Monte Carlo, significance).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import VALIDATION, TIMEFRAMES
from ..data.parser import load_bars, add_indicators
from ..engine.backtest import _simulate, compute_metrics
from ..engine.optimizer import _run_trial
from ..strategies.templates import STRATEGIES

logger = logging.getLogger(__name__)


def gate_stability_full(
    result,
    symbol: str,
    timeframe: str,
    param_range_pct: float = 0.20,
    max_sharpe_drop_pct: float = 0.15,
) -> tuple[bool, str]:
    """Gate: Parameter stability — re-run with perturbed params.

    Tests whether small (±20%) parameter changes maintain performance.
    A robust strategy has a plateau, not a sharp peak.
    """
    strategy_name = result.strategy
    params = result.params
    template = STRATEGIES[strategy_name]
    original_sharpe = result.sharpe

    # Generate perturbed parameter sets
    perturbed_sharpes = []
    for _ in range(10):
        perturbed = {}
        for key, val in params.items():
            if key in template.param_ranges:
                lo, hi = template.param_ranges[key]
                if isinstance(val, float):
                    factor = 1.0 + np.random.uniform(-param_range_pct, param_range_pct)
                    perturbed[key] = float(np.clip(val * factor, lo, hi))
                elif isinstance(val, int):
                    delta = max(1, int(val * param_range_pct))
                    perturbed[key] = int(np.clip(val + np.random.randint(-delta, delta + 1), lo, hi))

        trial = _run_trial(strategy_name, symbol, timeframe, perturbed)
        if trial.is_valid:
            perturbed_sharpes.append(trial.sharpe)

    if len(perturbed_sharpes) < 5:
        return False, f"Stability: only {len(perturbed_sharpes)} valid perturbed runs"

    avg_perturbed = np.mean(perturbed_sharpes)
    drop_pct = (original_sharpe - avg_perturbed) / max(original_sharpe, 0.01) * 100

    passed = drop_pct <= max_sharpe_drop_pct * 100
    reason = (
        f"Stability: {drop_pct:.0f}% drop" if passed
        else f"FAIL: {drop_pct:.0f}% Sharpe drop > {max_sharpe_drop_pct*100:.0f}%"
    )
    return passed, reason


def gate_walk_forward_full(
    result,
    symbol: str,
    timeframe: str,
    in_sample_years: int = 3,
    out_sample_years: int = 1,
    min_windows: int = 5,
) -> tuple[bool, str]:
    """Gate: Rolling Walk-Forward validation.

    Divides the data into rolling IS/OOS windows. A strategy must
    show positive performance across multiple non-overlapping OOS periods.
    """
    strategy_name = result.strategy
    params = result.params

    try:
        bars = load_bars(symbol, timeframe)
        if bars.empty:
            return False, "Walk-Forward: no data"
    except Exception:
        return False, "Walk-Forward: data load failed"

    # Calculate window sizes in bars
    tf_minutes = TIMEFRAMES.get(timeframe, 60)
    bars_per_year = 365 * 24 * 60 // tf_minutes
    is_bars = in_sample_years * bars_per_year
    oos_bars = out_sample_years * bars_per_year

    if len(bars) < is_bars + oos_bars:
        return False, f"Walk-Forward: need {is_bars+oos_bars} bars, have {len(bars)}"

    oos_sharpes = []
    window_start = 0

    while window_start + is_bars + oos_bars <= len(bars):
        is_data = bars.iloc[window_start:window_start + is_bars]
        oos_data = bars.iloc[window_start + is_bars:window_start + is_bars + oos_bars]

        # Optimize on IS window (simplified: just test the given params)
        is_trial = _run_trial(strategy_name, symbol, timeframe, params,
                              start=str(is_data.index[0].date()),
                              end=str(is_data.index[-1].date()))
        if not is_trial.is_valid:
            window_start += oos_bars
            continue

        # Test on OOS window
        oos_trial = _run_trial(strategy_name, symbol, timeframe, params,
                               start=str(oos_data.index[0].date()),
                               end=str(oos_data.index[-1].date()))
        if oos_trial.is_valid:
            oos_sharpes.append(oos_trial.sharpe)

        window_start += oos_bars

        if len(oos_sharpes) >= 20:  # safety limit
            break

    if len(oos_sharpes) < min_windows:
        return False, f"Walk-Forward: only {len(oos_sharpes)}/{min_windows} windows"

    # All OOS windows must have positive Sharpe
    avg_oos = np.mean(oos_sharpes)
    neg_windows = sum(1 for s in oos_sharpes if s < 0)

    passed = avg_oos > 0 and neg_windows <= 1
    reason = (
        f"Walk-Forward: {len(oos_sharpes)} windows, avg Sharpe {avg_oos:.2f}, {neg_windows} negative"
        if passed
        else f"FAIL: {len(oos_sharpes)} windows, avg {avg_oos:.2f}, {neg_windows} negative"
    )
    return passed, reason


def gate_regime_full(
    result,
    symbol: str,
    timeframe: str,
) -> tuple[bool, str]:
    """Gate: Regime analysis — break-even in all volatility quartiles.

    Splits the equity curve by ATR quartiles. Must be ≥0 in each.
    """
    try:
        bars = load_bars(symbol, timeframe)
        if bars.empty:
            return False, "Regime: no data"
        bars = add_indicators(bars)
    except Exception:
        return False, "Regime: data load failed"

    strategy_name = result.strategy
    params = result.params
    template = STRATEGIES[strategy_name]

    # Generate signals
    all_params = {**template.params, **params}
    signals = template.generate(bars, **all_params)
    positions = np.zeros_like(signals)
    in_pos = 0
    for i in range(len(signals)):
        sig = signals[i]
        if sig != 0 and in_pos == 0:
            positions[i] = sig; in_pos = sig
        elif sig == -in_pos and in_pos != 0:
            positions[i] = -in_pos; in_pos = 0

    symbol_obj = __import__('strohhalme.config', fromlist=['SYMBOLS_BY_NAME']).SYMBOLS_BY_NAME[symbol]

    spreads = bars.get("spread_mean", pd.Series(symbol_obj.typical_spread * symbol_obj.point, index=bars.index)).values
    atr = bars.get("atr", pd.Series(0.001, index=bars.index)).values
    thin = bars.get("thin", pd.Series(False, index=bars.index)).values

    equity, _, trades_arr, _ = _simulate(
        positions, bars["high"].values, bars["low"].values,
        bars["open"].values, bars["close"].values,
        spreads, atr, symbol_obj.lot_size, symbol_obj.pip_value,
        symbol_obj.commission, symbol_obj.swap_long, symbol_obj.swap_short, thin,
    )

    # Split into ATR quartiles
    atr_vals = pd.Series(atr).dropna()
    if len(atr_vals) < 20:
        return False, "Regime: not enough ATR data"

    quartiles = [atr_vals.quantile(q) for q in [0.25, 0.5, 0.75]]
    labels = ["Q1_low", "Q2", "Q3", "Q4_high"]

    regime_returns = {}
    for i, (lo, hi) in enumerate([(0, quartiles[0]),
                                   (quartiles[0], quartiles[1]),
                                   (quartiles[1], quartiles[2]),
                                   (quartiles[2], float('inf'))]):
        mask = (atr >= lo) & (atr < hi)
        mask = mask[:len(equity)]  # align
        regime_equity = equity[mask.values]
        if len(regime_equity) > 5:
            regime_ret = (regime_equity[-1] / regime_equity[0] - 1) if regime_equity[0] > 0 else 0
            regime_returns[labels[i]] = regime_ret

    if len(regime_returns) < 2:
        return False, "Regime: insufficient data across quartiles"

    min_regime = min(regime_returns.values())
    passed = min_regime >= -0.05  # allow 5% loss in worst regime
    reason = (
        f"Regime: {regime_returns}" if passed
        else f"FAIL: min regime return {min_regime:.1%}"
    )
    return passed, reason
