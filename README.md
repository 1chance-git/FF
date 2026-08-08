# FF — Statistical Arbitrage Trading System

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
├── tests/
│   ├── test_foundation_config.py # config.json + strategy sanity checks
│   ├── test_bot_startup.py       # full FreqtradeBot construction (network mocked)
│   ├── test_market_data.py       # market data layer unit tests
│   ├── test_regression.py        # rolling regression engine unit tests
│   ├── test_cointegration.py     # stat-arb engine unit tests
│   ├── test_risk.py              # risk engine unit tests
│   └── test_stat_arb_swing.py    # assembled strategy unit tests
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

### Next module

Hyperparameter tuning / hyperopt space and a dedicated backtest report —
not yet started.
