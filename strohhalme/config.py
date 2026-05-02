"""Configuration central — single source of truth for all pipeline params."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

ROOT = Path(os.environ.get("STROHHALME_ROOT", Path(__file__).parent.parent))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Symbol:
    name: str              # e.g. "EURUSD"
    pip_value: float       # value of 1 pip per 1 lot in USD
    point: float           # 1 point = e.g. 0.00001
    digits: int
    typical_spread: float  # in points
    swap_long: float       # per lot per day
    swap_short: float
    commission: float = 7.0  # per lot per round-turn (typical ECN)
    lot_size: int = 100_000


SYMBOLS = {
    "EURUSD": Symbol("EURUSD", 10.0, 0.00001, 5, 1.2, -8.0, 3.0),
    "GBPUSD": Symbol("GBPUSD", 10.0, 0.00001, 5, 1.8, -6.0, 2.0),
    "USDCHF": Symbol("USDCHF", 10.0 / 0.90, 0.00001, 5, 1.5, -5.0, 1.0),
    "USDJPY": Symbol("USDJPY", 10.0 / 150.0, 0.001, 3, 1.0, -4.0, 1.0),
    "EURGBP": Symbol("EURGBP", 10.0 * 1.25, 0.00001, 5, 1.5, -5.0, 2.0),
    "EURJPY": Symbol("EURJPY", 10.0 / 150.0, 0.001, 3, 1.8, -6.0, 2.0),
    "AUDUSD": Symbol("AUDUSD", 10.0, 0.00001, 5, 1.5, -3.0, 1.0),
    "USDCAD": Symbol("USDCAD", 10.0 / 1.38, 0.00001, 5, 1.8, -4.0, 1.0),
    "BTCUSD": Symbol("BTCUSD", 10.0, 1.0, 2, 10.0, 0.0, 0.0, lot_size=1),
    "ETHUSD": Symbol("ETHUSD", 10.0, 0.01, 2, 10.0, 0.0, 0.0, lot_size=1),
    "SP500": Symbol("SP500", 12.5, 1.0, 2, 1.0, 0.0, 0.0, commission=0, lot_size=1),
    "DAX": Symbol("DAX", 25.0, 1.0, 2, 1.0, 0.0, 0.0, commission=0, lot_size=1),
}
SYMBOLS_BY_NAME = {s.name: s for s in SYMBOLS.values()}

SYMBOLS_CORRELATED = [
    ("EURUSD", "GBPUSD"),
    ("EURUSD", "EURGBP"),
    ("USDCHF", "EURUSD"),   # negative correlation
]
SYMBOLS_UNCORRELATED = [
    ("EURUSD", "USDJPY"),
    ("EURUSD", "AUDUSD"),
]

TIMEFRAMES = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "M30": 30,
    "H1":  60,
    "H4":  240,
    "D1":  1440,
}

COST_MODEL: dict = {
    "spread_multiplier": 1.0,       # base multiplier for typical spread
    "news_spread_mult": 3.0,        # during high-impact news
    "rollover_spread_mult": 1.5,    # during 21:00-23:00 UTC
    "slippage_pct": 0.3,            # fraction of 15-min ATR as slippage
    "min_slippage_points": 0.5,     # minimum slippage in points
    "max_slippage_points": 50.0,
}

VALIDATION: dict = {
    "walk_forward": {
        "in_sample_years": 3,
        "out_sample_years": 1,
        "min_windows": 12,
        "fatal_if": "any_negative",  # or "mean_negative" or "pct_negative_gt_20"
    },
    "monte_carlo": {
        "shuffles": 1_000,
        "threshold_pct": 90.0,       # original must beat 90% of shuffles
    },
    "stability": {
        "param_range_pct": 20.0,     # test parameters ±20% around optimum
        "max_sharpe_drop_pct": 15.0, # plateau if drop < 15%
    },
    "cross_symbols": {
        "min_symbols": 3,
        "min_sharpe": 0.0,
    },
    "regime": {
        "atr_quartiles": [25, 50, 75],
        "min_break_even": True,       # must be ≥0 in every quartile
    },
    "significance": {
        "method": "holm_bonferroni",
        "alpha": 0.05,
    },
}

# Position sizing & risk management
# These determine how much capital is deployed per trade.
# Risk-based sizing ensures the account doesn't blow up from a single bad trade.
# Formula: position_size = equity * risk_per_trade / (atr * stop_atr)
RISK: dict = {
    "risk_per_trade": 0.01,          # 1% of current equity risked per trade
    "stop_atr": 2.0,                 # stop distance in ATR units
    "min_lot": 1_000,                # minimum lot (0.01 micro lot)
    "max_lot": 100_000,              # maximum lot (1.0 standard lot)
    "initial_equity": 10_000.0,      # starting account balance in USD
    "stop_out_pct": 0.50,            # stop trading if equity drops below this % of peak
}

PIPELINE: dict = {
    "max_strategies": 10_000,         # per run
    "max_parallel": 2,                # parallel backtests (cores - 2)
    "ram_limit_mb": 2_000,           # soft limit for data in memory
    "disk_limit_gb": 50,             # warning threshold
}
