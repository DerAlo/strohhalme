"""Main pipeline orchestrator.

Usage:
    python -m strohhalme.pipeline          # full run
    STROHHALME_ROOT=/data strohhalme       # custom data root
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import (
    ROOT, RESULTS, SYMBOLS_BY_NAME, SYMBOLS_CORRELATED, SYMBOLS_UNCORRELATED,
    PIPELINE, VALIDATION,
)
from .data.dukascopy import DukascopyDownloader
from .data.parser import ticks_to_bars, bars_to_parquet, load_bars
from .data.synthetic import ensure_data
from .data.yahoo import YahooDownloader
from .engine.optimizer import optimize
from .validation.gates import run_gates, all_passed
from .strategies.templates import STRATEGIES

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO):
    ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "pipeline.log"),
        ],
    )


async def download_data(
    symbols: list[str],
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    timeframes: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Download Dukascopy tick data → aggregate to bars → store as Parquet.

    Falls back to synthetic data if Dukascopy is unreachable (common from cloud IPs).

    Returns dict[symbol_tf_key] → DataFrame.
    """
    if timeframes is None:
        timeframes = ["H1"]

    all_bars: dict[str, pd.DataFrame] = {}

    dl = DukascopyDownloader(proxy_url=os.environ.get("DUKASCOPY_PROXY", "socks5://localhost:9050"))
    dukascopy_ok = False
    try:
        # Quick connectivity test through proxy
        import aiohttp
        from aiohttp_socks import ProxyConnector
        proxy_url = os.environ.get("DUKASCOPY_PROXY", "socks5://localhost:9050")
        connector = ProxyConnector.from_url(proxy_url)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as sess:
            async with sess.get(
                "https://datafeed.dukascopy.com/datafeed/EURUSD/2024/00/01/00h_ticks.bi5",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                dukascopy_ok = resp.status == 200
    except Exception:
        pass

    if dukascopy_ok:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        crypto_symbols = {"BTCUSD", "ETHUSD"}  # skip Dukascopy — crypto not available
        missing_symbols = []
        for symbol in symbols:
            if symbol in crypto_symbols:
                missing_symbols.append(symbol)
                continue
            logger.info("Downloading %s from %s to %s", symbol, start, end)
            ticks = await dl.download_range(symbol, start_dt, end_dt, progress=True)
            if ticks is not None and not ticks.empty:
                for tf in timeframes:
                    bars = ticks_to_bars(ticks, tf, symbol)
                    if not bars.empty:
                        bars_to_parquet(bars, symbol, tf)
                        all_bars[f"{symbol}_{tf}"] = bars
                        missing_symbols.remove(symbol) if symbol in missing_symbols else None
            else:
                missing_symbols.append(symbol)
        if missing_symbols:
            logger.warning("Dukascopy partial — downloading remaining %s via Yahoo...", missing_symbols)
            yahoo_dl = YahooDownloader()
            for symbol in missing_symbols:
                for tf in timeframes:
                    bars = await yahoo_dl.download_bars(symbol, start, end, tf)
                    if bars is not None and not bars.empty:
                        bars_to_parquet(bars, symbol, tf)
                        all_bars[f"{symbol}_{tf}"] = bars
    else:
        logger.warning("Dukascopy unreachable — trying Yahoo Finance...")
        yahoo_dl = YahooDownloader()
        yahoo_ok = False
        for symbol in symbols:
            for tf in timeframes:
                bars = await yahoo_dl.download_bars(symbol, start, end, tf)
                if bars is not None and not bars.empty:
                    bars_to_parquet(bars, symbol, tf)
                    all_bars[f"{symbol}_{tf}"] = bars
                    yahoo_ok = True
        if not yahoo_ok:
            logger.warning("Yahoo Finance also empty — using synthetic data")
            for symbol in symbols:
                for tf in timeframes:
                    bars = ensure_data(symbol, tf, start, end)
                    if not bars.empty:
                        all_bars[f"{symbol}_{tf}"] = bars

    await dl.close()
    return all_bars


def run_optimization(
    symbols: list[str],
    timeframes: list[str],
    start: str | None = None,
    end: str | None = None,
    n_samples: int = 50,
) -> list:
    """Run optimization → validation → ranking on existing data."""
    logger.info("Starting optimization: %d symbols × %d timeframes", len(symbols), len(timeframes))
    logger.info("Date range: %s → %s", start or "all", end or "all")

    results = optimize(
        strategies=None,    # all
        symbols=symbols,
        timeframes=timeframes,
        n_samples=n_samples,
        start=start,
        end=end,
    )

    logger.info("Optimization complete: %d valid results", len(results))

    # Run validation gates
    survivors: list[dict] = []
    for rank, result in enumerate(results[:100]):  # top 100 only
        outcomes = run_gates(result)
        passed = all_passed(outcomes)

        entry = {
            "rank": rank + 1,
            "strategy": result.strategy,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "params": result.params,
            "metrics": result.metrics,
            "gates": outcomes,
            "all_passed": passed,
        }
        survivors.append(entry)

        if passed:
            logger.info(
                "✓ #%d: %s/%s/%s Sharpe=%.2f DD=%.1f%% Trades=%d",
                rank + 1, result.strategy, result.symbol, result.timeframe,
                result.sharpe, result.max_dd * 100, result.n_trades,
            )

    # Save results
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS / f"optimization_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(survivors, f, indent=2, default=str)

    n_passed = sum(1 for s in survivors if s["all_passed"])
    logger.info("Saved %d results (%d passed all gates) → %s", len(survivors), n_passed, result_path)

    return survivors


def main():
    parser = argparse.ArgumentParser(description="Strohhalme — MQL5 EA discovery pipeline")
    parser.add_argument("--download", action="store_true", help="Download fresh data")
    parser.add_argument("--symbols", nargs="+", default=["EURUSD"], help="Symbols to process")
    parser.add_argument("--timeframes", nargs="+", default=["H1", "M15"], help="Timeframes")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--optimize-only", action="store_true", help="Skip download, run optimization only")
    parser.add_argument("--samples", type=int, default=50, help="Number of parameter samples per strategy")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    logger.info("=" * 60)
    logger.info("Strohhalme v0.1.0 — EA Discovery Pipeline")
    logger.info(f"Root: {ROOT}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info("=" * 60)

    import asyncio

    if args.download:
        asyncio.run(download_data(
            args.symbols,
            start=args.start or "2023-01-01",
            end=args.end or "2025-12-31",
            timeframes=args.timeframes,
        ))

    if args.optimize_only or args.download or True:  # always optimize if data exists
        run_optimization(args.symbols, args.timeframes, args.start, args.end, n_samples=args.samples)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
