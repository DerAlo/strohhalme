# Strohhalme 🥤

Automated MQL5 Expert Advisor discovery pipeline.

## Quick Start

```bash
pip install -r requirements.txt

# Download data + run optimization
python -m strohhalme.pipeline \
  --download \
  --symbols EURUSD GBPUSD \
  --timeframes H1 M15 \
  --start 2020-01-01 \
  --end 2024-12-31
```

## Docker

```bash
docker build -t strohhalme .
docker run -v $(pwd)/data:/data -v $(pwd)/results:/results strohhalme \
  --symbols EURUSD --timeframes H1
```

## Architecture

```
Dukascopy Ticks → Parquet OHLCV → Strategy Generator → Backtest → Gates → Ranking
                                      ▲                        │
                                      └── Parameter Optimizer ─┘
```

### Validation Gates
1. Min trades (≥30)
2. Max drawdown (≤24%)
3. Sharpe (≥0.5)
4. Profit factor (≥1.3)
5. Monte Carlo (beats 90% of shuffled trades)
6. Statistical significance (t-test)
7. Parameter stability (plateau check)
8. Regime analysis (break-even all quartiles)
9. Walk-forward (rolling IS/OOS)

## Resource Budget

- RAM: < 2GB for in-memory data
- Disk: ~5GB per symbol-year raw → ~200MB Parquet
- CPU: 2 cores used for parallel backtests

## Data Sources

Dukascopy free historical tick data (2003-present).
No API key required. Rate-limited to 1 request/3s.
