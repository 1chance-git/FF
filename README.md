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
│   └── data/
│       └── market_data.py        # market data loading/cleaning/validation/alignment
├── tests/
│   ├── test_foundation_config.py # config.json + strategy sanity checks
│   ├── test_bot_startup.py       # full FreqtradeBot construction (network mocked)
│   └── test_market_data.py       # market data layer unit tests
└── user_data/
    ├── config.json                    # committed, no secrets
    ├── config-private.json.example    # template — copy to config-private.json
    ├── strategies/
    │   └── FoundationStrategy.py      # no-op strategy skeleton
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

### Next module

Pair selection / cointegration screening for `BTC/USDC` and `ETH/USDC`
futures — not yet started.
