# Test suite

275 tests across five categories, selectable via `pytest -m <marker>`
(markers registered in `pyproject.toml`).

```bash
pip install -r requirements-dev.txt
pytest                    # everything (~30s)
pytest -m unit             # fast, isolated unit tests only (~13s)
pytest -m strategy         # strategy validation
pytest -m backtest         # full offline backtest runs (~15s)
pytest -m regression       # golden/pinned-value tests
pytest -m numerical        # cross-implementation & invariant checks
```

## Categories

### Unit tests (`-m unit`, 220 tests)

One file per module under test, testing that module's public API in
isolation with synthetic data:

| File | Covers |
|---|---|
| `test_foundation_config.py` | `user_data/config.json` structure, `FoundationStrategy` skeleton |
| `test_bot_startup.py` | Real `FreqtradeBot` construction against the project config (network mocked) |
| `test_market_data.py` | `stat_arb.data.market_data` — load/clean/align/validate OHLCV |
| `test_regression.py` | `stat_arb.signal.regression` — rolling OLS hedge ratio |
| `test_cointegration.py` | `stat_arb.signal.cointegration` — spread/z-score/cointegration gate |
| `test_risk.py` | `stat_arb.risk.risk` — stop loss, sizing, regime/trend, exposure, cooldown |
| `test_stat_arb_swing.py` | `StatArbSwing`'s pure helper functions and individual Freqtrade hooks |
| `test_hermes_*.py` | `hermes/` — logging, health checks, backtest launcher, process lifecycle, CLI |

### Strategy validation (`-m strategy`, `test_strategy_validation.py`, 12 tests)

Validates the *assembled* `StatArbSwing` strategy object against
Freqtrade's own interface contract — distinct from the per-hook unit
tests in `test_stat_arb_swing.py`:

* Freqtrade's real `validate_config_consistency` check passes for the
  committed config + strategy.
* Required `IStrategy` hooks, `can_short`, `INTERFACE_VERSION`,
  `stoploss`, `minimal_roi`, `startup_candle_count` are all structurally
  correct.
* **No-lookahead holds through the *composed* pipeline**: perturbing
  only the last few candles of input must not change any earlier
  `populate_indicators` → `populate_entry_trend` → `populate_exit_trend`
  output — the whole-strategy counterpart to the per-module lookahead
  tests in `test_regression.py`/`test_cointegration.py`, confirming
  composition didn't reintroduce lookahead at the seams (e.g. in the
  market-data alignment step).
* Entry signals are directionally mutually exclusive (never
  `enter_long` and `enter_short` on the same candle).

### Backtest validation (`-m backtest`, `test_backtest_validation.py`, 8 tests)

Runs Freqtrade's **real, complete backtesting engine** — not a mock of
it — against synthetic local candle data, with only the network
boundary (exchange market loading) mocked via
`tests.conftest.mocked_hyperliquid_exchange` (the same technique
`test_bot_startup.py` established). This validates that the assembled
system (config + `StatArbSwing` + all four `stat_arb` modules) produces
a coherent backtest result end to end, not just that each piece works in
isolation:

* The backtest completes and returns Freqtrade's real results structure.
* The engineered oscillating synthetic spread actually produces trades
  (not a tautological "it ran without crashing").
* **Both legs of the pair get traded** — validates the role-flipping
  entry/exit logic actually produces coordinated, opposite-direction
  positions on `ETH/USDC:USDC` and `BTC/USDC:USDC`.
* Every trade has finite, positive prices/amounts and respects the
  configured 5% stop loss.
* Position-sizing arithmetic (`amount * open_rate ≈ stake_amount *
  leverage`) reconciles.
* The same fixture data produces byte-identical trades across repeated
  runs (determinism check).

> This sandbox blocks outbound access to exchange APIs (see the root
> README), so a real network-backed backtest can't run here — the same
> limitation `test_bot_startup.py` documents for `freqtrade trade`. This
> suite proves the full backtesting engine works correctly by mocking
> only that one network boundary and running everything else for real.

### Regression tests (`-m regression`, `test_golden_values.py`, 6 tests)

Golden/pinned-value tests: a fixed, seeded synthetic dataset with the
*exact* expected output of each module recorded and asserted against.
Distinct from correctness tests elsewhere in the suite (which check
properties like "the estimated hedge ratio is close to the true beta")
— these catch **unintended behavior drift**: a future refactor that
subtly changes rolling-window edge handling, a default trend term, or a
threshold, in a way that shifts these exact numbers, even if the result
still "looks reasonable" to a property-based test.

### Numerical consistency (`-m numerical`, `test_numerical_consistency.py`, 29 tests)

Two kinds of checks:

* **Cross-implementation agreement** — the same quantity computed our
  way vs. an independent path (`numpy.linalg.lstsq` for the hedge
  ratio, `scipy.stats.linregress` for the trend filter, a direct
  `statsmodels.tsa.stattools.adfuller` call for regime detection) must
  agree to floating-point precision.
* **Arithmetic invariants** — identities that must hold by construction
  regardless of input (`units * price == notional`, `|entry - stop| /
  entry == stop_loss_pct`), and determinism (identical input must
  produce byte-identical output across repeated calls).

## Shared fixtures (`conftest.py`)

* `make_oscillating_cointegrated_pair` — synthetic cointegrated
  `(y, x)` OHLCV data with a deliberate sinusoidal spread deviation, so
  tests that need to observe actual trading behavior (not just "ran
  without error") get one reliably.
* `mocked_hyperliquid_exchange` — the network-boundary-only mock context
  manager used by the backtest-validation suite (and available for any
  future test needing a real, fully-offline Freqtrade engine instance).
* `fake_futures_market` — a minimal, complete ccxt-style futures market
  dict for the project's two pairs.
