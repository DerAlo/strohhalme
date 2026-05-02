#!/usr/bin/env python3
"""
Strohhalme Advanced Validation — Walk-Forward, Stabilität, Regime
Läuft auf Kandidaten die alle Basis-Gates bestanden haben.
"""
import json
import logging
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strohhalme.validation.advanced import (
    gate_stability_full,
    gate_walk_forward_full,
    gate_regime_full,
)
from strohhalme.engine.optimizer import TrialResult
from strohhalme.strategies.templates import STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def fix_param_types(strategy_name, params):
    """Convert string params from JSON to proper int/float types."""
    template = STRATEGIES.get(strategy_name)
    if not template:
        return params
    fixed = {}
    for key, val in params.items():
        if key in template.param_ranges:
            lo, hi = template.param_ranges[key]
            if isinstance(lo, int) or key in ("sma_fast", "sma_slow", "channel_period",
                                                "bb_period", "atr_period", "fast", "slow",
                                                "signal"):
                fixed[key] = int(val) if not isinstance(val, (int, np.integer)) else int(val)
            else:
                fixed[key] = float(val) if not isinstance(val, (float, np.floating)) else float(val)
        else:
            fixed[key] = val
    return fixed

# Lade aktuellste Optimierung
opt_files = sorted(RESULTS.glob("optimization_*.json"))
if not opt_files:
    log.error("Keine Optimierungs-Ergebnisse gefunden!")
    sys.exit(1)

latest = opt_files[-1]
with open(latest) as f:
    data = json.load(f)

candidates = [d for d in data if d.get("all_passed")]
log.info(f"Prüfe {len(candidates)} Kandidaten aus {latest.name}")

advanced_results = []
for c in candidates:
    log.info(f"\n{'='*60}")
    log.info(f"Kandidat #{c['rank']}: {c['strategy']}/{c['symbol']}/{c['timeframe']}")
    log.info(f"  Params: {c['params']}")
    log.info(f"  PF={c['metrics']['profit_factor']:.2f} DD={c['metrics']['max_drawdown']:.1%}")

    # Konvertiere Param-Typen (JSON speichert alles als String)
    params = fix_param_types(c['strategy'], c['params'])

    # Bau TrialResult aus dict
    result = TrialResult(
        strategy=c['strategy'],
        symbol=c['symbol'],
        timeframe=c['timeframe'],
        params=params,
        metrics=c['metrics'],
    )

    gates = {}

    # 1. Parameter-Stabilität
    try:
        passed, reason = gate_stability_full(result, c['symbol'], c['timeframe'])
        gates["stability"] = {"passed": passed, "reason": reason}
        icon = "✓" if passed else "✗"
        log.info(f"  {icon} Stability: {reason}")
    except Exception as e:
        gates["stability"] = {"passed": False, "reason": str(e)}
        log.info(f"  ⚠ Stability: {e}")

    # 2. Walk-Forward (aufwändig)
    try:
        passed, reason = gate_walk_forward_full(result, c['symbol'], c['timeframe'])
        gates["walk_forward"] = {"passed": passed, "reason": reason}
        icon = "✓" if passed else "✗"
        log.info(f"  {icon} Walk-Forward: {reason}")
    except Exception as e:
        gates["walk_forward"] = {"passed": False, "reason": str(e)}
        log.info(f"  ⚠ Walk-Forward: {e}")

    # 3. Regime-Analyse
    try:
        passed, reason = gate_regime_full(result, c['symbol'], c['timeframe'])
        gates["regime"] = {"passed": passed, "reason": reason}
        icon = "✓" if passed else "✗"
        log.info(f"  {icon} Regime: {reason}")
    except Exception as e:
        gates["regime"] = {"passed": False, "reason": str(e)}
        log.info(f"  ⚠ Regime: {e}")

    # Gesamt
    all_advanced_passed = all(g.get("passed", False) for g in gates.values())
    c["advanced_gates"] = gates
    c["advanced_passed"] = all_advanced_passed

    if all_advanced_passed:
        log.info(f"  ✅ ALLE ADVANCED GATES BESTANDEN — LIVE-READY!")
    else:
        failed = [k for k, v in gates.items() if not v.get("passed")]
        log.info(f"  ❌ {len(failed)} Gate(s) failed: {', '.join(failed)}")

    advanced_results.append(c)

# Speichern
result_path = RESULTS / f"validation_{latest.stem.replace('optimization_', '')}.json"
with open(result_path, "w") as f:
    json.dump(advanced_results, f, indent=2, default=str)

survivors = [c for c in advanced_results if c.get("advanced_passed")]
log.info(f"\n{'='*60}")
log.info(f"VALIDIERUNG ABGESCHLOSSEN")
log.info(f"  Geprüft: {len(advanced_results)} Kandidaten")
log.info(f"  Advanced-Gates bestanden: {len(survivors)}")
log.info(f"  Gespeichert: {result_path}")
for s in survivors:
    log.info(f"  ✅ {s['strategy']}/{s['symbol']}/{s['timeframe']} PF={s['metrics']['profit_factor']:.2f} DD={s['metrics']['max_drawdown']:.1%}")
