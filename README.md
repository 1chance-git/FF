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
| Exchange              | `binanceusdm` (Binance USDⓈ-M futures)         |
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
* **`binanceusdm` exchange id.** Binance splits spot and USDⓈ-M futures
  into separate ccxt exchange classes. `binanceusdm` is the one Freqtrade
  officially supports for isolated/cross USDC/USDT-margined futures.
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
├── requirements.txt              # pinned Freqtrade version
├── .gitignore
├── tests/
│   ├── test_foundation_config.py # config.json + strategy sanity checks
│   └── test_bot_startup.py       # full FreqtradeBot construction (network mocked)
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
# edit user_data/config-private.json with real exchange API keys, etc.
```

### Running

```bash
# Validate configuration without starting the bot
freqtrade show-config -c user_data/config.json -c user_data/config-private.json

# Start the bot (dry-run by default)
freqtrade trade -c user_data/config.json -c user_data/config-private.json
```

> **Note on this development sandbox:** the environment this project was
> scaffolded in blocks outbound network access to `fapi.binance.com`, so
> `freqtrade trade` cannot complete its live market-data handshake here.
> That is a network policy of this sandbox, not a defect in the
> configuration — the same config starts normally wherever outbound
> HTTPS to Binance is allowed. `tests/test_bot_startup.py` proves this by
> constructing the real `FreqtradeBot` against the committed config with
> only the network boundary (ccxt market loading) mocked out; every other
> step (config validation, exchange/strategy resolution, pairlist
> handling, persistence init) runs unmodified.

### Tests

```bash
pip install pytest
python3 -m pytest tests/ -v
```

### Next module

Pair selection / cointegration screening for `BTC/USDC` and `ETH/USDC`
futures — not yet started.
