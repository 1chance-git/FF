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
Historical market data  (Hyperliquid primary / Binance USD-M futures secondary, research-only)
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
| `Dockerfile` | Builds the research-worker image: installs `requirements.txt`, installs this repo (`pip install -e .`), runs the actual process as a non-root user. Default command is inert (`hermes --help`) — the real command is supplied per deploy, see below. |
| `docker-entrypoint.sh` | Runs first, as root, purely to `chown` the Railway Volume mounted at `/app/user_data/data` (Volumes are created empty and root-owned) before dropping to the non-root `hermes` user and exec'ing the real command. See "Live verification performed on Railway" for why this exists. |
| `.dockerignore` | Keeps the image to source code only. Historical data, logs, and Hermes' SQLite history are runtime state, not image contents — they belong on a Railway Volume. |
| `railway.json` | Tells Railway to build from the Dockerfile and run as a one-shot job (`restartPolicyType: NEVER` — a finished backtest should not be restarted in a loop). |
| `user_data/config-research.json` | **Primary** research-only Freqtrade config: same pairs and timeframe as the live config (`BTC/USDC:USDC`, `ETH/USDC:USDC`, `5m`, since `StatArbSwing.Y_PAIR`/`X_PAIR` are hardcoded to these two), and `exchange.name` is `hyperliquid` — the same exchange used live, reused here purely as a historical-data source. `dry_run` is always `true`; this file is never passed to `freqtrade trade`, only to `download-data`/`backtesting` (directly or via `hermes backtest`). Validated locally against Freqtrade's own `validate_config_consistency` — see "Local validation" below. |
| `user_data/config-research-binance.json` | **Secondary/validation** research-only config — same shape, `exchange.name` is `binanceusdm`. Used to cross-check backtest results against a second venue after running the same backtest against the primary (Hyperliquid) config; not the default, not used unless explicitly passed via `-c`. |

## Why Hyperliquid is primary and Binance is secondary/validation only

This module originally used Binance USD-M futures as the sole research
data source, reasoning that Hyperliquid was blocked in the local
sandbox (`403` on `CONNECT` to `api.hyperliquid.xyz`, matching every
other exchange host tried from that sandbox — a sandbox-wide network
policy, not evidence about any exchange specifically). Once actually
deployed to Railway, that assumption got tested for real:

- Railway's outbound network is genuinely unrestricted — confirmed by
  getting a real HTTP response back from Binance, not a connection
  block.
- That response was `451 Service unavailable from a restricted
  location`, per Binance's own `'b. Eligibility'` terms clause — a
  geo-restriction on the Railway region the service was deployed to
  (`us-west2`/`sfo`), not a network failure and not specific to this
  repo.

Rather than switch Railway's region to chase that block, or drop
Binance outright, the data-sourcing decision is: **Hyperliquid — the
exchange `StatArbSwing` already trades live on — is the primary
research data source** (`user_data/config-research.json`), since it was
never actually tested from an unblocked network and reusing one
exchange for both live trading and research is simpler than running
two. **Binance USD-M futures remains available as a secondary,
validation-only source** (`user_data/config-research-binance.json`) for
cross-checking results against a second venue, once/if its
geo-restriction from Railway's current region is no longer a blocker
(a different region, a different network path, etc.) — it was not
deleted, just demoted from default.

Both configs keep `StatArbSwing`'s hardcoded futures pairs
(`BTC/USDC:USDC` / `ETH/USDC:USDC`) and market shape (perpetual
futures, USDC-margined) so no strategy changes are needed to backtest
against either. **This pairing is purely a data-pipeline choice, not a
validation of the strategy itself** — see the note on Path A/B below.
The live exchange account is never touched by anything running on
Railway regardless of which research config is used.

## The two future research hypotheses (not yet buildable)

You asked the eventual framework to support testing two separate
hypotheses. Both are **documented here as the target, not implemented**
— `StatArbSwing` is a fixed two-leg mean-reversion pairs strategy for
exactly BTC/ETH, and neither hypothesis can run against it as-is:

- **Path A — trend** (`BTC/USDC`, `ETH/USDC`, `SOL/USDC`): three legs,
  not two. This isn't a pairs-trading question at all; it needs a
  different strategy shape than `StatArbSwing` provides. **This is
  understood to be the actual production direction** — the BTC/ETH
  pairing that `config-research.json`/`config-research-binance.json`
  use is a leftover of `StatArbSwing` being the only strategy that
  currently exists, not an endorsement of BTC/ETH mean reversion as a
  validated thesis. Nothing in this repo has tested that thesis; running
  a backtest against these configs proves the data pipeline works, not
  that the strategy is sound.
- **Path B — mean reversion** (`stETH/ETH`, `WBTC/BTC`, `cbETH/ETH`):
  these are staking-derivative/wrapped-asset *ratio* pairs, not
  USDC-margined perpetual futures — they may not exist as futures
  markets on Hyperliquid or Binance at all, and **which venue actually
  lists each leg has not been checked** (this document previously
  assumed neither exchange would have them; that assumption itself
  hasn't been verified either). The correct next step, once network
  access from an unblocked environment is confirmed working (see
  "Local validation" below), is per-pair venue discovery: call
  `ccxt.<exchange>().load_markets()` for each candidate exchange and
  check whether both legs of each Path B pair actually appear — not
  assumed to be Hyperliquid or Binance by default, and not assumed to
  be a single venue for all three pairs. Testing the cointegration
  hypothesis itself starts with the tooling that already exists
  (`stat_arb.signal.cointegration`, `stat_arb.risk.risk`) run offline
  against whatever price history is actually available, before any
  strategy is written around it — exactly as you specified ("do NOT
  assume Path B is cointegrated").

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
3. **Set the start command per run**, via the service's Settings →
   Deploy → Start Command in the Railway dashboard (or the equivalent
   API/MCP call). `railway.json` intentionally does **not** set a
   `deploy.startCommand` — Railway's own docs are explicit that a
   config-as-code file's settings always override dashboard/API
   settings for the same field, so if `railway.json` pinned a command,
   no dashboard or API change could ever take effect without editing
   and redeploying the file itself. Leaving it unset there is what
   makes "set it per run" actually true. The Dockerfile's own `CMD`
   (`hermes --help`) is still the safe fallback if no start command is
   set anywhere. The first reproducible job — reusing exactly the
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
   needs to exist under `user_data/data/hyperliquid/futures/` — via
   `freqtrade download-data -c user_data/config-research.json --timerange ...`,
   run once against the attached Volume.)

   To cross-check against the secondary (Binance) source instead, swap
   `-c user_data/config-research.json` for `-c
   user_data/config-research-binance.json` in either command — everything
   else stays the same.
4. **Read the result back**, either via `hermes analyze` (same command,
   same container) or by pulling `hermes_memory.sqlite3` off the Volume.

## Environment variables you'll eventually need

None of these are secrets in the usual sense — Hyperliquid's and
Binance's public OHLCV/market-metadata endpoints need no authentication
for `download-data`/`backtesting`, and this environment never places
orders. Nothing below was created, requested, or exposed by this
module; this is a list for when you configure the Railway service.

| Variable | Purpose | Required? |
|---|---|---|
| *(none)* | Public market-data endpoints (Hyperliquid primary, Binance secondary) require no API key for `download-data`/`backtesting`. | — |
| `HERMES_API_USERNAME` / `HERMES_API_PASSWORD` | Only relevant if you ever run `hermes health` against a *live* bot's REST API from Railway — unrelated to the research path above. Not needed for backtesting. | No |

If a future hypothesis genuinely needs authenticated data, add it as a
Railway service variable at that time — do not commit it to either
research config or any file in this repo, consistent with how
`user_data/config.json` already keeps Hyperliquid's
`walletAddress`/`privateKey` out of git via
`user_data/config-private.json` (gitignored).

## Live verification performed on Railway

Unlike the original version of this document (written before any real
deploy existed), this has now actually been deployed and exercised:

- **Docker build**: confirmed working — Railway builds this repo's
  `Dockerfile` successfully (had to fix the service's builder setting
  once; see below).
- **Volume**: created; originally mounted at `/app/user_data`, which
  shadowed the repo-baked config/strategy files (see "Volume shadowing"
  below) and was moved to `/app/user_data/data`.
- **Start command precedence bug, found and fixed**: setting a start
  command via the Railway dashboard/API had no effect — the container
  kept running the Dockerfile's inert default. Root cause: Railway's
  docs are explicit that config-as-code (`railway.json`) settings
  always override dashboard/API settings for the same field, and
  `railway.json` originally pinned `deploy.startCommand`. Fixed by
  removing `startCommand` from `railway.json` entirely, so the
  dashboard-set value now genuinely applies.
- **Volume shadowing repo files, found and fixed**: mounting the Volume
  directly at `/app/user_data` overlaid an empty directory over the
  baked-in `config-research.json`/`config.json`/`strategies/`, causing
  `Path 'user_data/config-research.json' does not exist`. Fixed by
  moving the mount path to `/app/user_data/data` (only the data
  subdirectory needs to persist across deploys).
- **Volume ownership permission error, found and fixed**: after the
  mount-path fix, the config file loaded correctly but
  `freqtrade backtesting` then failed with
  `PermissionError: [Errno 13] Permission denied: '/app/user_data/data/hyperliquid'`.
  Root cause: Railway Volumes are created empty and root-owned, but the
  container runs as the non-root `hermes` user (uid 1000) per the
  Dockerfile, so `hermes` couldn't create directories inside the mounted
  volume. Fixed by adding `docker-entrypoint.sh`: the container now
  starts as root (just long enough to `chown -R hermes:hermes
  /app/user_data/data`), then drops to the `hermes` user via `su` before
  exec'ing the actual start command — the process itself still never
  runs as root.
- **Binance USD-M futures reachability**: confirmed *blocked* — real
  HTTP `451` from `fapi.binance.com`, a Binance-side geo-restriction on
  Railway's deployed region (see "Why Hyperliquid is primary" above).
  This is what triggered demoting Binance to secondary/validation.
- **Hyperliquid reachability**: this is what this change (Hyperliquid as
  primary) is meant to test — still unverified as of writing this
  section, since every attempt so far failed before reaching the
  exchange call (first on the missing-config-file bug, then on the
  volume-permission error above). The next redeploy, with both fixes
  in place, is the actual test; check that deployment's logs for
  whether market/`exchangeInfo` loading succeeds or fails, and don't
  assume either outcome here.

## Local validation performed (before any Railway deploy existed)

- Both `user_data/config-research.json` and
  `user_data/config-research-binance.json` — valid JSON, and pass
  Freqtrade's own `validate_config_consistency()` after resolving
  `StatArbSwing` against them (the same check `freqtrade` runs before
  every real start, and the same one `tests/test_strategy_validation.py`
  already asserts for the live config).
- `railway.json` — valid JSON, matches Railway's documented schema
  shape.
- `Dockerfile` — reviewed by hand against `requirements.txt` /
  `pyproject.toml` before ever being built for real (now confirmed
  building successfully on Railway — see above).
- Full existing test suite (`pytest`) — run unchanged, to confirm this
  module didn't touch or break anything it shouldn't have.
