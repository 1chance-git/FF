# Production audit

Full review of the statistical arbitrage trading system across code
quality, architecture, performance, numerical correctness, memory usage,
error handling, logging, statistical assumptions, and risk controls.
Refactors were applied only where a concrete, justified issue was found;
each is described below with what was wrong, why it mattered, and how it
was verified fixed. Everything not listed as a finding was reviewed and
judged already sound.

All 358 tests pass after every change in this audit (up from 344 before
it — 14 new tests added specifically to lock in the fixes below). Full
suite runtime dropped from ~43s to ~22s as a direct consequence of the
performance fixes.

## Findings and fixes

### 1. Risk controls: `confirm_trade_entry` failed *open* on error (critical)

**Severity: critical.** Freqtrade calls `confirm_trade_entry` through
`strategy_safe_wrapper(..., default_retval=True)`
(`freqtradebot.py`) — meaning an *unhandled* exception anywhere inside it
gets logged and then treated as **entry confirmed**. `StatArbSwing`'s
`confirm_trade_entry` had no top-level exception handling, so any
unexpected failure in the risk engine, the dataprovider, or the wallet
lookup would silently *allow* the trade — the exact opposite of what a
risk gate exists to do.

**Fix:** wrapped the entire method body (moved to
`_confirm_trade_entry_unsafe`) in a blanket `try/except` that logs and
returns `False` on any exception, so the gate fails closed regardless of
what goes wrong inside it. Also hardened `self.wallets is None` handling
(previously silently coerced equity to `0.0`, which would then raise
deep inside `RiskEngine.evaluate_entry` instead of failing at the
strategy boundary with a clear log message).

`confirm_trade_exit` was reviewed under the same lens: Freqtrade also
defaults it to `True` on exception, which is the *safe* direction for an
exit (never get stuck unable to close a position) — but an exception
there was previously left to propagate as an alarming "Unexpected error"
log for what is really just cooldown bookkeeping. Wrapped
`risk_engine.record_exit` in its own `try/except` so a bookkeeping
failure is logged plainly and never risks the exit itself.

**Tests added** (`tests/test_stat_arb_swing.py`):
`test_strategy_confirm_trade_entry_fails_closed_on_unexpected_exception`,
`test_strategy_confirm_trade_entry_fails_closed_when_risk_engine_raises`,
`test_strategy_confirm_trade_entry_blocks_when_wallets_unavailable`,
`test_strategy_confirm_trade_exit_still_confirms_when_record_exit_raises`
— each injects a failure at a different point and asserts the gate's
fail-closed/fail-safe direction, not just its happy path.

### 2. Performance: regime detection's ADF search cost dominated runtime

**Severity: high** (live-trading latency risk). `detect_regime` runs one
`statsmodels.tsa.stattools.adfuller` call per rolling window via
`.rolling().apply()`. Profiling found this took **3.5 seconds** for a
1,000-row series at a 30-bar window — comparable to or exceeding
`user_data/config.json`'s 5-second `process_throttle_secs`, for what is
only one of several computations `populate_indicators` runs per pair,
per loop iteration, for two pairs.

The cost was almost entirely `adfuller`'s default `autolag="AIC"`
search, whose search range grows with window size
(`~12 * (nobs/100)**0.25` lags). For the 30-60 bar windows this project
actually uses, testing lags beyond 1-2 is both unreliable (each extra
lag consumes degrees of freedom from an already-small window) and,
empirically, the dominant runtime cost.

**Fix:** added `RegimeConfig.adf_maxlag` (default `2`), bounding
`autolag`'s search range without removing automatic lag selection
within that bound. Verified classification is unchanged on both a
synthetic mean-reverting and a trending series before/after the change.
Result: **3.5s → ~1.4s** for the same benchmark (2.5x).

**Tests added** (`tests/test_risk.py`):
`test_regime_config_default_adf_maxlag_is_bounded`,
`test_detect_regime_capped_maxlag_matches_uncapped_classification`;
updated `tests/test_numerical_consistency.py`'s
`test_adf_pvalue_matches_direct_statsmodels_call` to assert against the
same bounded call the implementation actually makes (it previously
compared against an unbounded call, which would have silently stopped
being an exact match once the bound was introduced) and added
`test_detect_regime_stays_fast_on_a_realistic_live_history_length`, a
wall-clock budget test guarding against this regressing silently.

### 3. Performance: `RiskEngine.evaluate_entry` recomputed over full history for a value it only reads once

**Severity: high**, compounding finding #2. `evaluate_entry` — called
once per prospective live trade via `confirm_trade_entry` — ran
`detect_regime`/`compute_trend_filter` over the *entire* `prices` series
passed in, even though it only ever reads `.iloc[-1]` (the latest
value) from each. A rolling window's value at the last row depends only
on the trailing `window` observations, so this recomputed thousands of
ADF calls to answer a question that needs exactly one.

**Fix:** trim `prices` to `prices.tail(window)` for each check before
calling `detect_regime`/`compute_trend_filter`, since a rolling
window's last-row output is provably identical either way (verified
empirically and via a dedicated test). Result: **1.42s → 0.003s**
(~450x) for a 1,000-row history at window=30; **the fix does not
change `populate_indicators`'s cost** (which correctly still needs the
full historical series for backtesting) — it's scoped to the
single-decision live-trading path.

**Tests added** (`tests/test_risk.py`):
`test_engine_evaluate_entry_decision_unaffected_by_history_length`
(long vs. short history produce an identical decision),
`test_engine_evaluate_entry_stays_fast_on_a_long_history` (wall-clock
budget guard on a 3,000-row history).

### 4. Memory usage: Hermes' JSON log file grew without bound

**Severity: medium.** `hermes.logging_config.configure_logging` used a
plain `logging.FileHandler` for the JSON log channel — no size cap, no
rotation — inconsistent with the trading bot's own `log_config` (see
`user_data/config.json`), which already uses a `RotatingFileHandler`
(10 MB × 10 backups) for exactly this reason. A long-lived operational
tool writing an ever-growing log file is a real disk-usage risk in
production.

**Fix:** switched to `logging.handlers.RotatingFileHandler`, with
`json_log_max_bytes`/`json_log_backup_count` added to `LoggingConfig`
(defaulting to the same 10 MB × 10 as the bot's own config), both
validated in `__post_init__`.

**Tests added** (`tests/test_hermes_logging.py`):
`test_json_log_file_handler_rotates_instead_of_growing_unbounded`,
`test_json_log_file_actually_rotates_when_size_limit_exceeded`,
`test_config_rejects_non_positive_max_bytes`,
`test_config_rejects_negative_backup_count`.

### 5. Security: committed REST API secrets looked like real values

**Severity: medium.** `user_data/config.json`'s `api_server.enabled` is
`false` by default and its `jwt_secret_key`/`ws_token` were always
meant to be placeholders overridden via the gitignored
`config-private.json` — but the committed values were plausible-looking
64-character hex strings, not obviously placeholders. An operator who
only flips `enabled: true` without ever creating `config-private.json`
would unknowingly run the API server with a JWT secret that's public in
git history, letting anyone with repo access forge authentication
tokens.

**Fix:** replaced both values with an unmistakable placeholder string
(`"INSECURE_PLACEHOLDER_COMMITTED_TO_GIT_REPLACE_VIA_CONFIG_PRIVATE_JSON"`,
still ≥32 chars to satisfy Freqtrade's schema) and added an inline
`//api_server_security` comment in the config explaining why and what
to do instead. Verified `freqtrade show-config` still validates cleanly.

### 6. Security: `hermes health --password` exposed secrets via CLI args

**Severity: low-medium.** The Freqtrade REST API password was only
accepted as a bare `--password` CLI flag — visible in shell history and
`ps` output on any shared or logged system.

**Fix:** added `envvar="HERMES_API_PASSWORD"` (and
`HERMES_API_USERNAME`) to the `hermes health` command's options, so
operators can supply credentials via environment variable instead; the
flag still works for interactive/scripted use where that's acceptable.

**Test added** (`tests/test_hermes_cli.py`):
`test_health_command_reads_credentials_from_environment`.

### 7. Code quality: duplicate `__post_init__` introduced mid-audit, caught before commit

While adding `LoggingConfig.json_log_max_bytes`/`json_log_backup_count`
validation (finding #4), an edit briefly left two `__post_init__`
definitions on the same dataclass (the second silently shadowing the
first, dropping the new validation). Caught by re-reading the file
immediately after editing, before running tests — a reminder that
editing dataclasses with generated `__post_init__` boilerplate is worth
a direct read-back, not just a test run (the duplicate would not have
caused any test to fail, since the second definition was a strict
subset of the first).

## Reviewed, no changes needed

* **`stat_arb/data/market_data.py`, `stat_arb/signal/regression.py`,
  `stat_arb/signal/cointegration.py`** — no correctness, performance, or
  error-handling issues found. Validation is thorough and raises early;
  lookahead-prevention design (trailing-only rolling windows, lagged
  hedge ratio) is sound and already directly tested.
* **`hermes/process.py`, `hermes/backtest.py`, `hermes/health.py`** —
  PID-file identity verification, graceful-then-forceful shutdown, and
  never-raise health checks all reviewed and confirmed correct.
* **`optimize/*.py`** — all statistics delegate to
  `freqtrade.data.metrics`/`freqtrade.optimize.hyperopt_loss`, no
  parallel reimplementation to drift; grid/random search and
  walk-forward window generation are pure and already unit-tested
  against known-answer synthetic cases.
* **`RiskEngine`'s exposure/cooldown logic** — correct; `check_exposure_limits`
  is called unconditionally even when an earlier check (cooldown,
  regime) already blocked the entry, which is mildly wasteful but O(1)
  and not worth the added branching complexity to skip.
* **Statistical assumptions** — Engle-Granger cointegration (`trend="c"`),
  fixed-fractional position sizing, and the regime/trend-filter split
  were all reviewed against their documented rationale in each module's
  docstring and found consistent with standard pairs-trading practice;
  no assumption was found undocumented or silently violated.
* **No `print()` in library code, no bare `except:`, no `SettingWithCopyWarning`/`FutureWarning`**
  surfaced across the full test suite run with warnings elevated.
