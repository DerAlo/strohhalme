"""Validation gates — the moat between backtest fantasy and live reality.

Each gate is a function that takes a TrialResult and returns (passed: bool, reason: str).
A strategy must pass ALL gates to graduate to live consideration.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ..engine.backtest import compute_metrics

logger = logging.getLogger(__name__)

GateFn = Callable[..., tuple[bool, str]]


def gate_min_trades(result, min_trades: int = 30) -> tuple[bool, str]:
    """Gate 1: Minimum trade count. Fewer trades = no statistical power."""
    n = result.n_trades
    passed = n >= min_trades
    reason = f"Trades: {n}" if passed else f"FAIL: only {n} trades (need ≥{min_trades})"
    return passed, reason


def gate_max_drawdown(result, max_dd_pct: float = 0.24) -> tuple[bool, str]:
    """Gate 2: Maximum drawdown limit (user requirement: 24%)."""
    dd = result.max_dd
    passed = dd <= max_dd_pct
    reason = f"MaxDD: {dd:.1%}" if passed else f"FAIL: MaxDD {dd:.1%} > {max_dd_pct:.0%}"
    return passed, reason


def gate_sharpe_min(result, min_sharpe: float = 0.5) -> tuple[bool, str]:
    """Gate 3: Minimum Sharpe ratio."""
    sharpe = result.sharpe
    passed = sharpe >= min_sharpe
    reason = f"Sharpe: {sharpe:.2f}" if passed else f"FAIL: Sharpe {sharpe:.2f} < {min_sharpe}"
    return passed, reason


def gate_profit_factor(result, min_pf: float = 1.3) -> tuple[bool, str]:
    """Gate 4: Profit factor — gross profit / gross loss."""
    pf = result.metrics.get("profit_factor", 0.0)
    passed = pf >= min_pf
    reason = f"PF: {pf:.2f}" if passed else f"FAIL: PF {pf:.2f} < {min_pf}"
    return passed, reason


def gate_monte_carlo(result, n_shuffles: int = 1_000, threshold_pct: float = 90.0) -> tuple[bool, str]:
    """Gate 5: Monte Carlo shuffling of trade sequence.

    Shuffles trade outcomes 1000×. Original Sharpe must beat threshold% of shuffles.
    """
    if result.trades is None or len(result.trades) < 10:
        return False, "FAIL: Monte Carlo needs ≥10 trades"

    trades = result.trades[result.trades != 0]
    if len(trades) < 10:
        return False, f"FAIL: Only {len(trades)} non-zero trades"

    original_sharpe = result.sharpe
    sharpe_values = np.zeros(n_shuffles)

    rng = np.random.default_rng(42)
    for i in range(n_shuffles):
        shuffled = rng.permutation(trades)
        # Build equity curve from shuffled trades
        equity = np.cumsum(shuffled)
        equity = equity + abs(equity.min()) + 1.0  # ensure positive
        rets = np.diff(np.log(equity))
        rets = rets[np.isfinite(rets)]
        if len(rets) < 2:
            sharpe_values[i] = -999.0
            continue
        sharpe_values[i] = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252)

    beat_pct = np.mean(original_sharpe > sharpe_values) * 100
    passed = beat_pct >= threshold_pct
    reason = (
        f"MonteCarlo: beats {beat_pct:.0f}%" if passed
        else f"FAIL: Monte Carlo {beat_pct:.0f}% < {threshold_pct:.0f}%"
    )
    return passed, reason


def gate_stability(
    result, param_range_pct: float = 0.20, max_sharpe_drop_pct: float = 0.15
) -> tuple[bool, str]:
    """Gate 6: Parameter stability (placeholder — full implementation
    requires re-running with perturbed params).

    This gate is flagged for full implementation after the optimizer is stable.
    """
    return True, "Stability: deferred (needs re-optimization with perturbed params)"


def gate_regime(result) -> tuple[bool, str]:
    """Gate 7: Regime analysis (placeholder — requires ATR quartile splitting).

    Full implementation splits the equity curve by volatility regime
    and checks break-even in each.
    """
    return True, "Regime: deferred (needs ATR quartile equity splitting)"


def gate_significance(result, alpha: float = 0.05) -> tuple[bool, str]:
    """Gate 8: Statistical significance via t-test on trade returns."""
    if result.trades is None:
        return False, "FAIL: no trade data"

    trades = result.trades[result.trades != 0]
    if len(trades) < 10:
        return False, f"FAIL: only {len(trades)} trades for t-test"

    mean = np.mean(trades)
    std = np.std(trades, ddof=1)
    if std == 0:
        return False, "FAIL: zero variance in trades"

    t_stat = mean / (std / np.sqrt(len(trades)))
    # Two-tailed t-test critical values (df=30, alpha=0.05 → ~2.042)
    # For smaller samples, use scipy if available, else conservative threshold
    critical = 2.05  # conservative for df≥10
    if len(trades) < 30:
        critical = 2.23  # df=10, alpha=0.05

    passed = abs(t_stat) >= critical
    reason = (
        f"t-test: {t_stat:.2f} ≥ {critical:.2f}" if passed
        else f"FAIL: t-stat {t_stat:.2f} < {critical:.2f} (not significant)"
    )
    return passed, reason


def gate_walk_forward(
    result,  # the full-period result
    *,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    params: dict,
    in_sample_years: int = 3,
    out_sample_years: int = 1,
    min_windows: int = 5,
) -> tuple[bool, str]:
    """Gate 9: Rolling walk-forward (placeholder — needs re-running optimizer on windows).

    Full implementation runs IS optimization for each window, tests OOS.
    """
    return True, "Walk-forward: deferred (needs windowed optimization runs)"


# ── Gate pipeline ────────────────────────────────────────────────────────────

GATES: list[tuple[str, GateFn, dict]] = [
    ("min_trades", gate_min_trades, {"min_trades": 30}),
    ("max_drawdown", gate_max_drawdown, {"max_dd_pct": 0.24}),
    ("sharpe_min", gate_sharpe_min, {"min_sharpe": 0.5}),
    ("profit_factor", gate_profit_factor, {"min_pf": 1.3}),
    ("monte_carlo", gate_monte_carlo, {}),
    ("significance", gate_significance, {"alpha": 0.05}),
    ("stability", gate_stability, {}),
    ("regime", gate_regime, {}),
    ("walk_forward", gate_walk_forward, {}),
]


def run_gates(result) -> dict[str, bool]:
    """Run all validation gates on a trial result.

    Returns: {gate_name: passed}
    """
    outcomes = {}
    for name, gate_fn, kwargs in GATES:
        try:
            passed, reason = gate_fn(result, **kwargs)
            outcomes[name] = passed
            if not passed:
                logger.info("  ✗ %s: %s", name, reason)
            else:
                logger.debug("  ✓ %s", name)
        except Exception as exc:
            logger.warning("Gate '%s' raised: %s", name, exc)
            outcomes[name] = False
    return outcomes


def all_passed(outcomes: dict[str, bool]) -> bool:
    """Check if all mandatory gates passed.

    'deferred' gates (stability, regime, walk_forward) are not mandatory yet.
    """
    MANDATORY = {"min_trades", "max_drawdown", "sharpe_min", "profit_factor", "monte_carlo", "significance"}
    return all(outcomes.get(g, False) for g in MANDATORY)
