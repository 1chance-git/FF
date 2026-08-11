# Railway research environment

Railway hosts **research and backtesting infrastructure only**. It never
runs `freqtrade trade`, never holds exchange credentials for the live
account, and Freqtrade/Hermes' own logic on Railway is byte-for-byte
identical to what already runs locally — nothing about the strategy,
risk engine, or Hermes changes because it's running on Railway instead
of a laptop.

```
LOCAL DEVELOPMENT
    |
    v
GitHub repository
    |
    v
RAILWAY RESEARCH ENVIRONMENT  (this document)
    |
    v
Historical market data  (Binance USD-M futures, research-only)
    |
    v
Freqtrade backtests  (StatArbSwing, unchanged)
    |
    v
Results  (BacktestResult)
    |
    v
Hermes memory/analysis  (hermes_memory.sqlite3, hermes analyze)
```

Live trading (Hyperliquid, `user_data/config.json`) is a **separate,
untouched path** that never crosses this one. Nothing here changes
`exchange.name`, `trading_mode`, `margin_mode`, the live pair
whitelist, or any strategy/risk parameter.

## What's in this repo for Railway

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the research-worker image: installs `requirements.txt`, installs this repo (`pip install -e .`), runs as a non-root user. Default command is inert (`hermes --help`) — the real command is supplied per deploy, see below. |
| `.dockerignore` | Keeps the image to source code only. Historical data, logs, and Hermes' SQLite history are runtime state, not image contents — they belong on a Railway Volume. |
| `railway.json` | Tells Railway to build from the Dockerfile and run as a one-shot job (`restartPolicyType: NEVER` — a finished backtest should not be restarted in a loop). |
| `user_data/config-research.json` | A **research-only** Freqtrade config: same pairs and timeframe as the live config (`BTC/USDC:USDC`, `ETH/USDC:USDC`, `5m`, since `StatArbSwing.Y_PAIR`/`X_PAIR` are hardcoded to these two), but `exchange.name` is `binanceusdm` — a historical-data source only. `dry_run` is always `true`; this file is never passed to `freqtrade trade`, only to `download-data`/`backtesting` (directly or via `hermes backtest`). Validated locally against Freqtrade's own `validate_config_consistency` — see "Local validation" below. |

## Why Binance USD-M futures, not Hyperliquid, for data

Hyperliquid is blocked in the current sandbox by an outbound network
policy (confirmed via `curl` and `hermes backtest` in earlier sessions —
`403` on the `CONNECT` tunnel to `api.hyperliquid.xyz`, and the same for
every Binance host tried from that sandbox too). Railway is expected to
have a normal outbound network path, but the blocker we hit wasn't
Hyperliquid-specific — it may not be reachable everywhere. Binance
USD-M futures (`binanceusdm`) is used here strictly as a **candle data
source**, so that:

- `StatArbSwing`'s hardcoded futures pairs (`BTC/USDC:USDC` /
  `ETH/USDC:USDC`) keep the same market shape (perpetual futures,
  USDC-margined) the strategy already assumes — no strategy changes
  needed to backtest against this data.
- The live exchange (Hyperliquid) is never touched by anything running
  on Railway.

If Binance also turns out to be unreachable from wherever Railway
actually runs, that's a data-source choice to revisit — not a reason to
change the live exchange, which this module does not touch under any
circumstance.

## The two future research hypotheses (not yet buildable)

You asked the eventual framework to support testing two separate
hypotheses. Both are **documented here as the target, not implemented**
— `StatArbSwing` is a fixed two-leg mean-reversion pairs strategy for
exactly BTC/ETH, and neither hypothesis can run against it as-is:

- **Path A — trend** (`BTC/USDC`, `ETH/USDC`, `SOL/USDC`): three legs,
  not two. This isn't a pairs-trading question at all; it needs a
  different strategy shape than `StatArbSwing` provides.
- **Path B — mean reversion** (`stETH/ETH`, `WBTC/BTC`, `cbETH/ETH`):
  these are staking-derivative/wrapped-asset *ratio* pairs, not
  USDC-margined perpetual futures — they wouldn't exist as futures
  markets on Binance or Hyperliquid at all. Testing this hypothesis
  starts with the cointegration/ADF tooling that already exists
  (`stat_arb.signal.cointegration`, `stat_arb.risk.risk`) run offline
  against spot/ratio price history, before any strategy is written
  around it — exactly as you specified ("do NOT assume Path B is
  cointegrated").

Building either is out of scope for this module (no new strategy, no
FreqAI, no two-leg execution changes). What Railway gives you once
deployed is a reproducible place to *run that offline statistical
testing* — the existing `stat_arb` modules already have OLS regression,
hedge-ratio estimation, ADF testing, Engle-Granger cointegration, and
rolling z-scores; half-life, transaction-cost/funding-cost/slippage
modeling, and a Path-A/B-aware strategy do not exist yet and weren't
added here, per "do not implement these yet unless they already exist."

## What you'll need to do once Railway access is available

1. **Create one Railway service** from this GitHub repo (branch
   `claude/stat-arb-trading-system-fslf82`, or whatever it's merged
   into) — Railway will detect `railway.json` and build via the
   committed `Dockerfile` automatically.
2. **Attach a Volume** to that service, mounted at `/app/user_data`
   (or at minimum `/app/user_data/data` and
   `/app/user_data/hermes_memory.sqlite3`'s parent directory). Without
   this, downloaded candle data and Hermes' trade/backtest history
   would be wiped on every redeploy — a Volume is what makes runs
   reproducible and lets Hermes memory accumulate across them.
3. **Set the start command per run**, overriding the safe default in
   `railway.json`. The first reproducible job — reusing exactly the
   `hermes backtest` path already built and tested — is:

   ```
   python -m hermes backtest \
     -c user_data/config-research.json \
     --strategy StatArbSwing \
     --strategy-path user_data/strategies \
     --timerange 20240101-20240111 \
     --user-data-dir user_data \
     --timeout 300
   ```

   (Before this will produce trades, the research data itself still
   needs to exist under `user_data/data/binanceusdm/futures/` — via
   `freqtrade download-data -c user_data/config-research.json --timerange ...`,
   run once against the attached Volume. Neither command has been run
   here; both were blocked by the sandbox's network policy in the prior
   session, not by anything in this repo.)
4. **Read the result back**, either via `hermes analyze` (same command,
   same container) or by pulling `hermes_memory.sqlite3` off the Volume.

## Environment variables you'll eventually need

None of these are secrets in the usual sense — Binance's public OHLCV
endpoints need no authentication, and this environment never places
orders. Nothing below was created, requested, or exposed by this
module; this is a list for when you configure the Railway service.

| Variable | Purpose | Required? |
|---|---|---|
| *(none)* | `binanceusdm` public market-data endpoints require no API key for `download-data`/`backtesting`. | — |
| `HERMES_API_USERNAME` / `HERMES_API_PASSWORD` | Only relevant if you ever run `hermes health` against a *live* bot's REST API from Railway — unrelated to the research path above. Not needed for backtesting. | No |

If a future hypothesis genuinely needs authenticated data (e.g. a
higher-rate-limit Binance API key), add it as a Railway service
variable at that time — do not commit it to `user_data/config-research.json`
or any file in this repo, consistent with how `user_data/config.json`
already keeps Hyperliquid's `walletAddress`/`privateKey` out of git via
`user_data/config-private.json` (gitignored).

## Local validation performed

Docker itself could not be exercised end-to-end in the sandbox this was
built in — the `docker` CLI is present but no daemon is running
(`docker info` fails with "cannot connect to the Docker daemon"), so no
actual `docker build` was possible here. What *was* validated locally:

- `user_data/config-research.json` — valid JSON, and passes Freqtrade's
  own `validate_config_consistency()` after resolving `StatArbSwing`
  against it (the same check `freqtrade` runs before every real start,
  and the same one `tests/test_strategy_validation.py` already asserts
  for the live config).
- `railway.json` — valid JSON, matches Railway's documented schema
  shape (`build.builder`/`build.dockerfilePath`,
  `deploy.startCommand`/`deploy.restartPolicyType`).
- `Dockerfile` — reviewed by hand against `requirements.txt` /
  `pyproject.toml` (base image Python version matches
  `requires-python = ">=3.11"`; installs the same dependencies local
  dev uses, no extras invented); could not be built without a daemon.
- Full existing test suite (`pytest`) — run unchanged, to confirm this
  module didn't touch or break anything it shouldn't have.

**A real `docker build .` should be run** (locally with Docker Desktop,
in CI, or via Railway's own build step) before relying on this image —
that step was not possible here.
