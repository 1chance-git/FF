# FF — Statistical Arbitrage Trading System

> **Production audit:** see [`AUDIT.md`](AUDIT.md) for a full review of
> code quality, architecture, performance, numerical correctness, memory
> usage, error handling, logging, statistical assumptions, and risk
> controls across the whole system, including one critical fix (a risk
> gate that failed *open* rather than closed on error) and two
> significant live-trading performance fixes.

## Foundation module

This is the project foundation: a clean, launchable [Freqtrade](https://www.freqtrade.io/)
installation with **no trading logic yet**. It exists to prove that the
exchange, account, and risk-configuration decisions for the whole system are
correct before any strategy code is written.

### What's configured

| Decision            | Value                                         |
|----------------------|-----------------------------------------------|
| Trading mode         | `futures`                                     |
| Margin mode           | `isolated`                                    |
| Stake currency        | `USDC`                                        |
| Exchange              | `hyperliquid` (Hyperliquid perpetuals DEX)      |
| Pairlist               | `StaticPairList` restricted to `BTC/USDC:USDC`, `ETH/USDC:USDC` |
| Default mode          | `dry_run: true` (paper trading, no real orders) |

See `user_data/config.json` for the full configuration and inline `//`
comments explaining each non-obvious choice.

### Design decisions

* **Freqtrade over a custom engine.** Freqtrade is a mature, actively
  maintained open-source trading framework with a battle-tested exchange
  abstraction (via `ccxt`), backtester, risk controls, and persistence
  layer. Building any of that from scratch would duplicate work a large
  open-source project has already solved well.
* **`hyperliquid` exchange id.** Hyperliquid is a fully on-chain perpetuals
  DEX with a mature, officially-supported Freqtrade/ccxt integration
  (`freqtrade.exchange.hyperliquid.Hyperliquid`). It natively margins
  perpetuals in USDC and supports both isolated and cross margin, so it
  fits this project's USDC/isolated requirements directly rather than
  needing a USDC-margined product bolted onto a USDT-native exchange.
* **Wallet-based credentials, not an API key/secret pair.** Hyperliquid
  authenticates orders by signing them with a wallet private key rather
  than issuing exchange-side API keys (`ccxt.hyperliquid`'s
  `requiredCredentials` are `walletAddress` + `privateKey`, not
  `apiKey`/`secret`). `user_data/config.json` declares both fields blank;
  real values are supplied only through the gitignored
  `config-private.json`, exactly like the other secrets below.
* **`StaticPairList` only.** Later modules (pair selection, cointegration
  screening) will decide *which* pairs to trade statistically. Until that
  logic exists, the pairlist is hard-restricted to exactly the two pairs
  this system is designed around, so no other market can accidentally be
  traded.
* **`dry_run: true` by default.** The foundation module's job is to prove
  the bot starts and is wired correctly — not to place real orders. Live
  trading requires deliberately overriding this in `config-private.json`
  (see below).
* **Secrets split into `config-private.json`.** API keys, Telegram
  tokens, and the REST API server's JWT/auth credentials never belong in
  version control. `user_data/config.json` (committed) contains no
  secrets; `user_data/config-private.json` (gitignored) holds them
  locally. Copy `user_data/config-private.json.example` to get started.
* **`log_config` (dictConfig) instead of bare `--logfile`.** Freqtrade
  supports a full `logging.config.dictConfig`-style block in the config
  file. This gives structured, rotating file logs
  (`user_data/logs/freqtrade.log`, 10 MB × 10 backups) plus console
  output, without needing extra CLI flags at every invocation.
* **`FoundationStrategy` is an intentional no-op.** It implements the
  three mandatory `IStrategy` hooks (`populate_indicators`,
  `populate_entry_trend`, `populate_exit_trend`) but they return the
  dataframe unchanged — no indicators, no signals. `can_short = True` is
  set now because futures short-selling is central to statistical
  arbitrage pair trades, even though no signal logic exists yet.

### Project layout

```
FF/
├── pyproject.toml                # stat_arb package metadata + pytest config
├── requirements.txt              # pinned Freqtrade version
├── .gitignore
├── stat_arb/                     # project code, independent of user_data/
│   ├── data/
│   │   └── market_data.py        # market data loading/cleaning/validation/alignment
│   ├── signal/
│   │   ├── regression.py         # rolling OLS hedge-ratio engine
│   │   └── cointegration.py      # spread/z-score + cointegration validation
│   └── risk/
│       └── risk.py               # independent risk engine (no Freqtrade dependency)
├── hermes/                       # operational tooling (own CLI, decoupled from stat_arb)
│   ├── logging_config.py         # structured JSON + rich console logging
│   ├── health.py                 # health checks via freqtrade_client
│   ├── backtest.py               # backtest launcher (subprocess)
│   ├── process.py                # process lifecycle + restart support
│   └── cli.py                    # `hermes` CLI (click + rich)
├── optimize/                     # optimization framework (own CLI, decoupled from stat_arb)
│   ├── hyperopt_loss.py          # custom IHyperOptLoss (Sharpe - drawdown penalty)
│   ├── hyperopt_launcher.py      # `freqtrade hyperopt` launcher (subprocess)
│   ├── grid_search.py            # Freqtrade-independent grid/random parameter search
│   ├── walk_forward.py           # rolling train/test window generation + orchestration
│   ├── reporting.py              # performance report (freqtrade.data.metrics + rich)
│   └── cli.py                    # `optimize-cli` CLI (click + rich)
├── tests/
│   ├── README.md                       # test suite organization, categories, fixtures
│   ├── conftest.py                     # shared fixtures (synthetic data, mocked exchange)
│   ├── test_foundation_config.py       # config.json + strategy sanity checks
│   ├── test_bot_startup.py             # full FreqtradeBot construction (network mocked)
│   ├── test_market_data.py             # market data layer unit tests
│   ├── test_regression.py              # rolling regression engine unit tests
│   ├── test_cointegration.py           # stat-arb engine unit tests
│   ├── test_risk.py                    # risk engine unit tests
│   ├── test_stat_arb_swing.py          # assembled strategy unit tests
│   ├── test_hermes_logging.py          # hermes logging unit tests
│   ├── test_hermes_health.py           # hermes health check unit tests
│   ├── test_hermes_backtest.py         # hermes backtest launcher unit tests
│   ├── test_hermes_process.py          # hermes process lifecycle unit tests
│   ├── test_hermes_cli.py              # hermes CLI unit tests
│   ├── test_strategy_validation.py     # assembled-strategy interface + no-lookahead validation
│   ├── test_backtest_validation.py     # real, fully offline end-to-end backtest runs
│   ├── test_golden_values.py           # regression tests: pinned golden values
│   ├── test_numerical_consistency.py   # cross-implementation + invariant checks
│   ├── test_hyperopt_loss.py           # optimize hyperopt loss function unit tests
│   ├── test_hyperopt_launcher.py       # optimize hyperopt launcher unit tests
│   ├── test_grid_search.py             # optimize grid/random search unit tests
│   ├── test_walk_forward.py            # optimize walk-forward unit tests
│   ├── test_reporting.py               # optimize performance reporting unit tests
│   └── test_optimize_cli.py            # optimize CLI unit tests
└── user_data/
    ├── config.json                    # committed, no secrets
    ├── config-private.json.example    # template — copy to config-private.json
    ├── strategies/
    │   ├── FoundationStrategy.py      # no-op strategy skeleton
    │   └── StatArbSwing.py            # assembled pairs-trading strategy
    ├── data/                          # downloaded candle data (gitignored)
    ├── logs/                          # rotating log files (gitignored)
    ├── backtest_results/
    ├── hyperopt_results/
    ├── hyperopts/
    ├── freqaimodels/
    ├── notebooks/
    └── plot/
```

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp user_data/config-private.json.example user_data/config-private.json
# edit user_data/config-private.json with your real wallet address/private key, etc.
```

### Running

```bash
# Validate configuration without starting the bot
freqtrade show-config -c user_data/config.json -c user_data/config-private.json

# Start the bot (dry-run by default)
freqtrade trade -c user_data/config.json -c user_data/config-private.json
```

> **Note on this development sandbox:** the environment this project was
> scaffolded in blocks outbound network access to exchange APIs, so
> `freqtrade trade` cannot complete its live market-data handshake with
> Hyperliquid here. That is a network policy of this sandbox, not a
> defect in the configuration — the same config starts normally wherever
> outbound HTTPS to Hyperliquid is allowed. `tests/test_bot_startup.py`
> proves this by constructing the real `FreqtradeBot` against the
> committed config with only the network boundary (ccxt market loading)
> mocked out; every other step (config validation, exchange/strategy
> resolution, pairlist handling, persistence init) runs unmodified.

### Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

See [`tests/README.md`](tests/README.md) for the full test suite —
unit, strategy validation, backtest validation, regression (golden
values), and numerical consistency tests, each selectable via
`pytest -m <marker>`.

## Market data module

`stat_arb.data.market_data` loads, cleans, validates, and aligns OHLCV
candle data for `BTC/USDC:USDC` and `ETH/USDC:USDC`. It contains **no
indicators or trading logic** — that is deliberately out of scope for
this module.

### What it does

* **Loads** candle data from Freqtrade's on-disk candle store
  (`user_data/data/hyperliquid/`) via `MarketDataLoader`, one pair or
  several at once.
* **Handles missing candles** by delegating to Freqtrade's own
  `clean_ohlcv_dataframe` / `ohlcv_fill_up_missing_data`: gaps are
  filled with the previous close (zero volume), the standard convention
  for "no trades happened in this interval."
* **Aligns timestamps** across pairs by taking the intersection of
  available timestamps (`align_pairs`) — required before any cross-pair
  statistic (spread, correlation, cointegration) can be computed, since
  those assume row *i* of both series is the same point in time.
* **Validates dataframe integrity** (`validate_ohlcv`): required
  columns present, no null/duplicate/unsorted timestamps, no
  null/negative/non-positive values, and the OHLC relationship
  (`low <= open, close <= high`) holds for every row. All violations
  found are reported together, not just the first.
* **`MarketDataService`** composes all of the above into one call:
  load → clean & fill → validate → align → validate again.

### Design decisions

* **Reuses Freqtrade's history/converter code instead of reimplementing
  it.** `freqtrade.data.history.load_pair_history` already reads the
  exact on-disk format `freqtrade download-data` and the live bot both
  write. `freqtrade.data.converter.clean_ohlcv_dataframe` already
  de-duplicates and gap-fills OHLCV data using logic exercised by a
  large production user base. This module is a thin, typed,
  validated orchestration layer on top of both — reimplementing either
  would duplicate mature code for no benefit.
* **Loading and processing are separated.** `MarketDataLoader` is the
  only piece that touches disk. `clean_and_fill`, `validate_ohlcv`, and
  `align_pairs` are pure functions over in-memory dataframes, so unit
  tests exercise gap-filling, validation, and alignment logic directly
  with synthetic data — no filesystem or exchange dependency required
  for the core logic.
* **Validation raises, it doesn't warn.** Silently accepting a corrupt
  dataframe (duplicate timestamps, an OHLC violation, an unfilled gap)
  would let bad data flow into statistical models downstream, where
  it's much harder to trace back to its source.
* **Alignment is an inner join, not a forward-fill across pairs.**
  Forward-filling one pair's price to cover a timestamp the other pair
  has no data for would fabricate a data point. Taking the intersection
  of timestamps instead guarantees both series in a pair only ever share
  timestamps that both genuinely have data for.

### Usage

```python
from pathlib import Path

from freqtrade.enums import CandleType
from stat_arb.data.market_data import MarketDataLoader, MarketDataService

loader = MarketDataLoader(
    datadir=Path("user_data/data/hyperliquid"),
    timeframe="5m",
    candle_type=CandleType.FUTURES,
)
service = MarketDataService(loader)

data = service.get_aligned_market_data(["BTC/USDC:USDC", "ETH/USDC:USDC"])
# data["BTC/USDC:USDC"] and data["ETH/USDC:USDC"] are cleaned, gap-filled,
# validated DataFrames sharing an identical `date` index.
```

Candle data must be downloaded first, e.g.:

```bash
freqtrade download-data -c user_data/config.json \
    -p BTC/USDC:USDC ETH/USDC:USDC -t 5m --trading-mode futures
```

> This sandbox blocks outbound access to exchange APIs (see the note
> above), so `download-data` cannot run here. `tests/test_market_data.py`
> covers the loader/service against locally-written synthetic candle
> files, so the module is fully verified without needing live data.

## Rolling regression engine

`stat_arb.signal.regression` fits a rolling OLS regression across two
aligned price series (e.g. the outputs of `MarketDataService`) and
produces a dynamic hedge ratio — the beta a pairs trader uses to size
the offsetting leg of a spread trade at each point in time. It contains
**no spread/z-score/signal-generation logic** — that is deliberately out
of scope for this module.

### What it does

* **Rolling OLS via `statsmodels`.** `RollingRegressionEngine.fit(y, x)`
  runs `statsmodels.regression.rolling.RollingOLS` on
  `y = intercept + hedge_ratio * x` over a configurable window,
  producing a `RollingRegressionResult` with a full `hedge_ratio` series
  (one estimate per timestamp, not a single static beta).
* **Validates numerical stability per window.** Each window's design
  matrix condition number is computed (closed-form, vectorized — see
  `_rolling_condition_number`) and combined with a sanity bound on the
  hedge ratio's magnitude. Windows failing either check are flagged
  `is_stable=False`; their hedge ratio is forward-filled from the last
  stable estimate by default (`ffill_unstable=True`), never silently
  trusted as-is.
* **Logs hedge ratio changes.** Every relative change in the (stable)
  hedge ratio at or above `hedge_ratio_change_log_threshold` (default
  5%) is logged at `INFO`, along with a `WARNING` summary whenever any
  window is flagged numerically unstable.

### Design decisions

* **`statsmodels.RollingOLS` does the regression math**, not a hand-rolled
  windowed `numpy.linalg.lstsq` loop — it is mature, correctly handles
  window edges and missing data, and updates incrementally rather than
  refitting from scratch every step.
* **Condition number is computed analytically from rolling sums**, not
  via a per-window `numpy.linalg.cond` call in a Python loop. For the
  two-column design matrix `[1, x]`, `X'X` is a 2x2 matrix built from
  `x.rolling().sum()` and `(x**2).rolling().sum()`, whose eigenvalues
  (and hence the 2-norm condition number) have a closed form — verified
  in the tests to match `numpy.linalg.cond` exactly, at a fraction of the
  cost.
* **Unstable windows are forward-filled, not dropped.** Downstream
  consumers need a hedge ratio at every timestamp; a stale-but-trustworthy
  estimate is safer than a numerically meaningless one. Callers who want
  raw `NaN` gaps instead can set `ffill_unstable=False`.
* **Only significant hedge-ratio changes are logged**, not every
  window-to-window fluctuation — logging every step of a rolling
  statistic would flood the log with routine noise instead of surfacing
  the regime shifts a pairs trader actually needs to know about.
* **The engine is pair-agnostic.** It doesn't know or care which pair
  leg is `y` vs. `x` — that assignment is a decision for the (not yet
  built) pair-selection/signal-generation module.

### Usage

```python
from stat_arb.signal.regression import RollingRegressionEngine, RollingRegressionConfig

engine = RollingRegressionEngine(RollingRegressionConfig(window=60))
result = engine.fit(y=btc_close, x=eth_close)

result.hedge_ratio       # dynamic hedge ratio, unstable windows forward-filled
result.is_stable         # per-window stability flag
result.condition_number  # per-window design-matrix condition number
result.n_unstable        # count of windows without a stable estimate
```

## Statistical arbitrage engine

`stat_arb.signal.cointegration` computes the pair spread and its rolling
mean/standard deviation/z-score from a price pair and a hedge ratio
(typically the output of `RollingRegressionEngine`), and validates that
the pair is actually cointegrated before treating any of that as a
trading signal input. It contains **no entry/exit rule logic** — that is
deliberately out of scope for this module.

### What it does

* **Computes the spread**: `spread = y - hedge_ratio * x`.
* **Rolling mean, standard deviation, and z-score** of the spread, all
  via trailing (never centered) `pandas` rolling windows.
* **Cointegration validation** via the Engle-Granger test
  (`statsmodels.tsa.stattools.coint`). `CointegrationEngine.compute()`
  raises `CointegrationError` by default if the pair fails the test —
  a spread built on a non-cointegrated pair isn't mean-reverting, and a
  z-score computed on it isn't a meaningful signal.
* **Prevents lookahead bias** at both of its possible entry points:
  rolling statistics are always trailing (`center=False`, not exposed as
  a toggle), and the hedge ratio is lagged (`hedge_ratio_lag`, default
  1 bar) before being applied to the spread, so the hedge ratio used at
  bar *t* was estimated using no information from bar *t* itself.

### Design decisions

* **Cointegration testing uses `statsmodels.tsa.stattools.coint`**
  (Engle-Granger), a mature, well-tested implementation — reimplementing
  the underlying augmented Dickey-Fuller regression and critical-value
  tables would duplicate established statistical code for no benefit.
* **Cointegration validation gates spread computation by default.** A
  non-cointegrated pair's spread is just noise dressed up as a signal;
  `require_cointegration=True` (the default) refuses to produce one.
  Set it to `False` for research/exploration where fitting anyway is
  useful.
* **The hedge ratio is lagged even though the rolling regression is
  already trailing.** A trailing regression window ending at bar *t*
  technically uses no *future* data, but it does include bar *t*'s own
  price — and since OLS explicitly minimizes each window's total squared
  residual, bar *t*'s own residual is disproportionately shrunk by a fit
  that was partly optimized around it. That in-sample shrinkage makes
  the spread look more mean-reverting than an honest out-of-sample
  hedge ratio would. Lagging the hedge ratio by (by default) one bar
  removes this specific bias; the module logs a warning if a caller
  explicitly sets `hedge_ratio_lag=0`.
* **A degenerate rolling standard deviation produces `NaN`, not `inf`.**
  Dividing by a standard deviation that has collapsed to numerical noise
  would produce a huge, meaningless z-score; `zscore_std_floor` treats
  those windows as unavailable data instead.
* **Lookahead prevention is directly unit-tested, not just asserted in
  a docstring.** `test_rolling_zscore_unaffected_by_future_perturbation`
  and similar tests perturb a *future* observation and assert every
  *earlier* output is byte-for-byte unchanged — the concrete, falsifiable
  form of "no lookahead bias."

### Usage

```python
from stat_arb.signal.cointegration import CointegrationEngine, CointegrationConfig

engine = CointegrationEngine(CointegrationConfig(spread_window=60, hedge_ratio_lag=1))
result = engine.compute(y=btc_close, x=eth_close, hedge_ratio=hedge_ratio)

result.spread          # y - lagged_hedge_ratio * x
result.rolling_mean     # trailing rolling mean of the spread
result.rolling_std      # trailing rolling standard deviation of the spread
result.zscore           # trailing rolling z-score — the signal input
result.cointegration    # CointegrationTestResult (Engle-Granger)
```

## Risk engine

`stat_arb.risk.risk` is an **independent risk-decision layer with no
dependency on Freqtrade** — every function and class operates on plain
`pandas`/`numpy`/primitive-Python types, so it can be tested, reused from
a backtester, or driven by a different execution engine entirely without
ever importing `freqtrade`. It contains six independent controls,
composed by `RiskEngine.evaluate_entry()` into one allow/deny decision:

* **Stop loss** — fixed 5% adverse-move stop
  (`compute_stop_loss_price` / `is_stop_loss_triggered`).
* **Position sizing** — fixed-fractional sizing from the stop distance,
  capped by `max_position_pct_of_equity` and `max_leverage`
  (`calculate_position_size`).
* **Regime detection** — classifies each rolling window as
  mean-reverting or trending via the Engle-Granger pair's own
  Augmented Dickey-Fuller test, applied locally (`detect_regime`).
* **Trend filter** — a faster, separate check: the statistical
  significance of a rolling linear time-trend slope, via
  `statsmodels.RollingOLS` (`compute_trend_filter`).
* **Maximum exposure** — gross, per-pair, and position-count limits
  (`check_exposure_limits`).
* **Cooldown logic** — blocks re-entry on a pair for a configurable
  period after a stop-loss exit (`CooldownTracker`).

### Design decisions

* **No Freqtrade dependency, by requirement.** The engine is a pure
  decision layer — given prices, equity, and current positions, it
  decides whether and how large an entry should be — kept free of
  `freqtrade` imports so it can be exercised in isolation and reused
  anywhere. A unit test parses the module's AST to enforce this
  directly (`test_risk_module_does_not_import_freqtrade`), not just by
  convention.
* **Regime detection reuses `statsmodels.tsa.stattools.adfuller`**, the
  same ADF implementation that validates cointegration elsewhere in this
  project, rather than a custom stationarity heuristic — applied per
  rolling window (documented as more expensive than a closed-form
  indicator; intended for periodic checks, not tick-by-tick).
* **The trend filter reuses `statsmodels.RollingOLS`** (already a
  project dependency) to regress price on a linear time index and test
  the slope's significance, rather than a hand-rolled moving-average
  crossover — giving a statistically grounded, cheap answer that is
  deliberately independent of (and can disagree with) the ADF-based
  regime classification.
* **Position sizing is capped by explicit exposure bounds, not just risk
  amount.** Sizing purely from the stop distance can produce an
  unbounded position when the stop is very tight; every size is also
  capped by `max_position_pct_of_equity` and `max_leverage`, with the
  binding cap reported on the result rather than applied silently.
* **`CooldownTracker` is the one deliberately mutable class in the
  module.** Every other config here is a frozen dataclass; a cooldown
  tracker's entire job is to remember exit history across calls, so
  immutability would defeat its purpose. Its state is scoped to exactly
  what it needs: `pair -> last qualifying exit time`.
* **Only stop-loss exits arm the cooldown by default.** A trade closed
  for a normal reason isn't evidence the pair or timing was bad; a
  stop-loss exit is. `CooldownConfig(apply_to_all_exits=True)` opts into
  cooling down after every exit instead.

### Usage

```python
from datetime import datetime, timezone
from stat_arb.risk.risk import RiskEngine, RiskEngineConfig, PositionSide

engine = RiskEngine(RiskEngineConfig())
decision = engine.evaluate_entry(
    pair="BTC/USDC:USDC",
    side=PositionSide.LONG,
    entry_price=50_000.0,
    equity=10_000.0,
    prices=spread_series,       # recent spread/price history
    current_positions={},        # pair -> current notional
    current_time=datetime.now(timezone.utc),
)

if decision.allowed:
    ...  # place the order sized at decision.position_size.units
engine.record_exit("BTC/USDC:USDC", exit_time, stop_loss_triggered=True)
```

## Assembled strategy: StatArbSwing

`user_data/strategies/StatArbSwing.py` is the Freqtrade `IStrategy`
that assembles all four modules above into a live, tradable pairs
strategy for the project's two-pair universe (`ETH/USDC:USDC` as the
"y" leg, `BTC/USDC:USDC` as the "x"/hedge leg). **No AI/ML component is
used anywhere** — every signal and decision comes from the closed-form
statistical and rule-based logic already built and tested in the four
modules; a unit test parses the strategy file's imports to enforce that
no ML framework (`freqAI`, `torch`, `sklearn`, etc.) is ever pulled in.

### How the modules compose

* **`populate_indicators()`** — fetches the *other* leg's OHLCV via the
  dataprovider, cleans/aligns both legs
  (`stat_arb.data.market_data.clean_and_fill` / `align_pairs` /
  `validate_ohlcv`), fits the rolling hedge ratio
  (`stat_arb.signal.regression.RollingRegressionEngine`), computes the
  spread/z-score with an Engle-Granger cointegration check
  (`stat_arb.signal.cointegration.CointegrationEngine`), and classifies
  the spread's regime and trend (`stat_arb.risk.risk.detect_regime` /
  `compute_trend_filter`) — merging all of it back onto Freqtrade's
  per-pair dataframe without changing its row count.
* **`populate_entry_trend()`** — enters long/short based on the z-score
  threshold, direction flipped depending on whether the current pair is
  the "y" or "x" leg (so both legs get coordinated, opposite-direction
  signals for the same spread event), gated by the regime, trend, and
  cointegration flags computed above.
* **`populate_exit_trend()`** — exits once the z-score has reverted
  toward the mean past a (smaller) exit threshold.
* **Risk engine integration beyond the vectorized hooks** —
  `custom_stake_amount` sizes every entry via
  `stat_arb.risk.risk.calculate_position_size`; `confirm_trade_entry`
  applies the risk engine's *stateful* controls (cooldown, portfolio
  exposure) that need live wallet/position state unavailable during
  vectorized backtesting; `confirm_trade_exit` arms the cooldown via
  `RiskEngine.record_exit` whenever `exit_reason == "stop_loss"`. The
  base `stoploss = -0.05` class attribute and the risk engine's
  `StopLossConfig` share one `STOP_LOSS_PCT` constant so they can't
  drift out of sync.

### Design decisions

* **Every real decision is a pure, freqtrade-independent function.**
  `determine_pair_role`, `build_aligned_closes`, `compute_pair_indicators`,
  `compute_entry_signals`, `compute_exit_signals`, and
  `positions_notional_from_trades` all take/return plain
  `pandas`/primitive types and are unit-tested directly — `StatArbSwing`
  itself is a thin adapter wiring Freqtrade's hooks to them.
* **Both legs are traded, driven by the same spread.** Genuine pairs
  trading needs both legs executed with opposite exposure. Since
  Freqtrade calls each hook once per pair, this strategy computes the
  *same* spread/z-score inside each call and flips entry/exit direction
  based on which leg is currently being evaluated.
* **Cointegration is validated once per indicator refresh, not per row.**
  Engle-Granger is a whole-sample test by construction — it isn't meant
  to be re-run per historical bar. Its result is broadcast across the
  refresh and naturally updates as new data arrives on each subsequent
  call.
* **Regime detection and the trend filter run on the spread, not raw
  price** — that's the actual question this strategy needs answered:
  is the *spread* mean-reverting right now, or trending away from it.
* **Two complementary gating layers.** `populate_entry_trend` carries
  the vectorized statistical filters Freqtrade's backtester can
  evaluate historically; `confirm_trade_entry` adds the stateful checks
  (cooldown, exposure) that only make sense with live portfolio state.
  Both read from the same `RiskEngine` instance, so there's one source
  of truth for every threshold.

### Usage

```bash
# Backtest (once historical BTC/USDC and ETH/USDC futures data is downloaded)
freqtrade backtesting -c user_data/config.json --strategy StatArbSwing

# Dry-run
freqtrade trade -c user_data/config.json -c user_data/config-private.json \
    --strategy StatArbSwing
```

## Hermes: operational tooling

`hermes/` is a standalone CLI and library for operating the bot process
this project runs — structured logging, health checks, a backtest
launcher, and process lifecycle/restart support — kept independent of
Freqtrade's internals and installed as its own `hermes` console command
(`pip install -e .`).

### What it does

* **Structured logging** (`hermes.logging_config`) — two independent
  output channels on the same `logging` hierarchy: JSON logs (via
  `python-json-logger`) for machine consumption, and colorized
  human-readable console output (via `rich`) for interactive use.
  Idempotent — re-configuring doesn't stack duplicate handlers.
* **Health checks** (`hermes.health`) — `HealthChecker` queries a
  running bot's REST API through `freqtrade_client.FtRestClient` (the
  official client library Freqtrade itself ships) and reports
  `api_reachable`, `bot_health`, and `resources` (CPU/RAM via the bot's
  own `/sysinfo`) as an aggregate `HealthReport` — `HEALTHY`,
  `DEGRADED`, or `UNHEALTHY`, never raising even if the bot is
  completely unreachable.
* **Backtest launcher** (`hermes.backtest`) — `BacktestLauncher` builds
  and runs `python -m freqtrade backtesting ...` as a subprocess,
  returning a structured `BacktestResult` (exit code, stdout/stderr,
  duration) rather than raising on a failed backtest.
* **Restart support** (`hermes.process`) — `BotProcessManager` tracks
  the bot via a PID file, `start`/`stop`/`restart`/`status`, with
  graceful `SIGTERM`-then-`SIGKILL` shutdown; `RestartSupervisor` adds a
  bounded, exponentially-backed-off retry loop (2s → 4s → 8s → 16s
  capped) for unattended crash recovery.
* **CLI-friendly output** (`hermes.cli`) — a `click`-based CLI
  (`hermes health` / `backtest` / `start` / `stop` / `restart` /
  `status`) rendering `rich` tables and colored status text, with
  process exit codes that reflect outcome (e.g. `hermes health` exits
  `1` if unhealthy) for scripting.

### Design decisions

* **Every mature library is reused, not reimplemented.** `FtRestClient`
  for the API protocol, `python-json-logger` for JSON formatting,
  `rich` for terminal rendering, `click` for CLI parsing, `psutil` for
  cross-platform process management — each is exactly the kind of
  fiddly-to-get-right, already-solved problem this project's other
  modules also avoid reinventing (e.g. `statsmodels` for regression and
  cointegration).
* **Freqtrade is managed as an external process, never imported and run
  in-process.** Both the backtest launcher and the process manager
  build an argv and hand it to `subprocess`, exactly the same
  arm's-length relationship `StatArbSwing.py` has with `stat_arb`'s pure
  functions — Freqtrade version upgrades that change CLI flags only
  require touching the (thin, tested) command-builder functions.
* **Command construction and command execution are always separate
  functions.** `build_backtest_command`/`build_trade_command` are pure
  (given a config, return the exact argv) and unit-tested without any
  subprocess; `BacktestLauncher.run`/`BotProcessManager.start` are the
  thin I/O wrappers — the same pure-vs-I/O separation used throughout
  this project.
* **A PID file's process identity is verified, not assumed.**
  `BotProcessManager` confirms a PID file's process is actually a
  freqtrade process (configurable via an injectable matcher, used by
  tests to substitute a lightweight stand-in) before trusting it,
  so a stale PID file pointing at an unrelated, since-reused PID
  correctly reports "not running" instead of a false positive.
* **Health checks never raise.** An unreachable bot is a normal,
  expected state (starting up, mid-restart, not started yet) — it's
  reported as `UNHEALTHY` with a clear reason, not an exception a caller
  has to catch.

### Usage

```bash
pip install -e .   # registers the `hermes` console command

hermes health --api-url http://127.0.0.1:8080 --username user --password pass
hermes backtest -c user_data/config.json --strategy StatArbSwing --timerange 20240101-20240401
hermes start -c user_data/config.json --strategy StatArbSwing
hermes status -c user_data/config.json --strategy StatArbSwing
hermes stop -c user_data/config.json --strategy StatArbSwing

# Structured JSON logs alongside the console output:
hermes --json-log-file user_data/logs/hermes.log start -c user_data/config.json --strategy StatArbSwing
```

## Optimization framework

`optimize/` provides hyperopt configuration, parameter search, walk-
forward testing, and performance reporting for `StatArbSwing` — built
alongside `hermes/` (both treat Freqtrade as an external process to
launch, never something to reimplement) but scoped specifically to
tuning and validating the strategy's parameters rather than operating
the live bot.

### What it does

* **Hyperopt configuration.** `StatArbSwing.py` now exposes
  `entry_zscore_param`/`exit_zscore_param` as Freqtrade
  `DecimalParameter`s (in the standard `buy`/`sell` spaces), and
  `optimize/hyperopt_loss.py` defines `StatArbHyperOptLoss` — a custom
  `IHyperOptLoss` combining Sharpe ratio and max drawdown (both via
  `freqtrade.data.metrics`) into one risk-adjusted objective.
  `optimize/hyperopt_launcher.py`'s `HyperoptLauncher` builds and runs
  `freqtrade hyperopt` as a subprocess, wiring `--hyperopt-path` at the
  package itself so Freqtrade finds the loss function automatically.
* **Parameter search.** `optimize/grid_search.py` provides `grid_search`/
  `random_search` — a lightweight, **Freqtrade-independent** sweep over
  any objective function, for quick coarse searches or parameters
  outside Freqtrade's hyperopt space entirely (e.g. sweeping
  `stat_arb.risk.risk.RiskEngineConfig` fields directly).
* **Walk-forward testing.** `optimize/walk_forward.py`'s
  `generate_windows` produces rolling train/test date-range pairs;
  `WalkForwardRunner` orchestrates optimize-on-train /
  evaluate-on-test across them via injected callables, reporting every
  window's out-of-sample score (not just an average) so regime-
  dependent overfitting shows up rather than averaging out.
* **Performance reporting.** `optimize/reporting.py`'s
  `compute_performance_report` assembles Sharpe/Sortino/Calmar/max
  drawdown/expectancy/SQN — every one a direct call into
  `freqtrade.data.metrics`, never reimplemented — into one
  `PerformanceReport`, rendered as a `rich` table matching `hermes`'s
  CLI output style.

### Design decisions

* **Freqtrade's own hyperopt engine (Optuna-based) does the actual
  parameter search**, not a custom optimizer — `HyperoptLauncher`
  launches the real `freqtrade hyperopt` CLI exactly the way
  `hermes.backtest.BacktestLauncher` launches `freqtrade backtesting`:
  build an argv, run it as a subprocess, return a structured result.
* **Hyperopt parameters are deliberately scoped to entry/exit
  thresholds only, not window sizes.** Tuning `REGRESSION_WINDOW` et al.
  would require Freqtrade's more expensive "indicator space" hyperopt
  mode and reconstructing `risk_engine` (and its stateful
  `CooldownTracker`) every epoch — added complexity and live-trading
  risk not worth it over tuning just the thresholds, which Freqtrade's
  standard `buy`/`sell` spaces already handle cheaply and safely.
* **`grid_search`/`random_search` take an injected objective function
  and know nothing about Freqtrade at all** — the same
  dependency-injection pattern `hermes.health.HealthChecker` uses —
  keeping the search algorithm itself fully unit-testable against a
  synthetic objective (a quadratic bowl with a known optimum),
  independent of whether a real backtest can run in the current
  environment.
* **Walk-forward's optimizer/evaluator are injected callables, not
  hardcoded to `HyperoptLauncher`/`hermes.backtest.BacktestLauncher`.**
  This makes `WalkForwardRunner`'s orchestration — window sequencing,
  per-window result aggregation — testable with fast synthetic
  stand-ins, while production callers wire in the real launchers.
* **Every reported statistic is a direct call into
  `freqtrade.data.metrics`.** Sharpe/Sortino/Calmar/drawdown/expectancy/
  SQN each have real subtleties (annualization convention, drawdown
  base) Freqtrade's own implementations already get right; a parallel
  reimplementation would only risk silently drifting from them.

### Usage

```bash
pip install -e .   # registers the `optimize-cli` console command

optimize-cli hyperopt -c user_data/config.json --strategy StatArbSwing --epochs 100
optimize-cli report --trades-file trades.json --starting-balance 10000
```

```python
from datetime import date, timedelta
from optimize.walk_forward import generate_windows, WalkForwardRunner

windows = generate_windows(
    date(2024, 1, 1), date(2024, 7, 1),
    train_period=timedelta(days=60), test_period=timedelta(days=14),
)
report = WalkForwardRunner(optimizer=my_optimizer, evaluator=my_evaluator).run(windows)
print(report.mean_out_of_sample_score, report.worst_window)
```

## Testing suite

344 tests across five categories — full details, per-category
breakdown, and shared fixtures are documented in
[`tests/README.md`](tests/README.md). Summary:

* **Unit tests** (`-m unit`, 286 tests) — one file per module, testing
  its public API in isolation with synthetic data.
* **Strategy validation** (`-m strategy`, 15 tests) — the *assembled*
  `StatArbSwing` against Freqtrade's own config/strategy consistency
  checks, plus no-lookahead validated through the composed
  `populate_indicators` → `populate_entry_trend` → `populate_exit_trend`
  pipeline (not just within each `stat_arb` module feeding it).
* **Backtest validation** (`-m backtest`, 8 tests) — Freqtrade's real
  backtesting engine, run fully offline against synthetic local data
  with only the network boundary mocked, validating the assembled
  system produces a coherent result: both pair legs actually trade,
  position sizing reconciles, the stop loss is respected, and repeated
  runs are deterministic.
* **Regression tests** (`-m regression`, 6 tests) — golden/pinned
  numeric values on a fixed seeded dataset, guarding against
  unintended behavior drift that a property-based correctness test
  wouldn't catch.
* **Numerical consistency** (`-m numerical`, 29 tests) — cross-checks
  against independent implementations (`numpy.linalg.lstsq`,
  `scipy.stats.linregress`, direct `statsmodels` calls) and arithmetic
  invariants (`units * price == notional`, determinism).

### Design decisions

* **The backtest-validation suite runs Freqtrade's real engine, not a
  simplified stand-in.** Mocking only the network boundary (exchange
  market loading — the same technique `test_bot_startup.py`
  established for constructing a real `FreqtradeBot`) and running
  everything else for real is strictly more trustworthy than a
  hand-rolled backtest simulator: every config-validation rule,
  every accounting detail, every interaction between Freqtrade and the
  strategy's hooks is the actual code that runs in production.
* **Synthetic data for backtest validation is deliberately engineered,
  not just seeded-random.** `make_oscillating_cointegrated_pair` adds a
  sinusoidal deviation on top of the cointegrating relationship so the
  spread reliably crosses entry/exit thresholds — a backtest that
  "completes successfully" while silently producing zero trades would
  pass a weaker test suite without ever exercising the entry/exit
  logic at all.
* **Golden-value tests are a distinct category from correctness
  tests, on purpose.** A test asserting "the hedge ratio is close to
  the true beta" and a test asserting "the hedge ratio at this exact
  point is 1.846089405611601" catch different failure modes — the
  first catches broken statistics, the second catches *any* behavior
  change at all, intentional or not, forcing it to be reviewed rather
  than silently shipped.
* **Numerical consistency checks lean on independent libraries, not
  just internal re-derivation.** Cross-checking against
  `scipy.stats.linregress` and direct `statsmodels` calls (rather than
  only against our own helper functions written differently) is
  stronger evidence of correctness, since a shared bug between our
  wrapper and a hand-rolled numpy check wouldn't be caught by the
  numpy check either.

### Next module

Deployment automation (CI/CD, remote monitoring dashboards) — not yet
started.
