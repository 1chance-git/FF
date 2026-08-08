"""Optimization framework: hyperopt configuration, parameter search, walk-forward testing, reporting.

Four complementary tools:

* `optimize.hyperopt_loss` / `optimize.hyperopt_launcher` — configures
  and launches Freqtrade's own (Optuna-based) hyperopt engine against
  `StatArbSwing`'s tunable entry/exit z-score thresholds.
* `optimize.grid_search` — a lightweight, Freqtrade-independent grid/
  random parameter search over an arbitrary objective function, for
  quick sweeps or parameters outside Freqtrade's hyperopt space.
* `optimize.walk_forward` — rolling train/test window generation and
  orchestration, to validate that optimized parameters generalize
  out-of-sample rather than merely fitting the data they were tuned on.
* `optimize.reporting` — performance statistics (Sharpe/Sortino/Calmar/
  drawdown/expectancy/SQN, all via `freqtrade.data.metrics`) and
  `rich`-rendered reports, shared by the other three.

See each submodule's docstring for its own design rationale.
"""

from optimize.grid_search import (
    Objective,
    ParameterCombination,
    ParameterSpec,
    SearchError,
    SearchResult,
    Trial,
    grid_search,
    random_search,
)
from optimize.hyperopt_launcher import (
    HyperoptConfig,
    HyperoptError,
    HyperoptLauncher,
    HyperoptResult,
    build_hyperopt_command,
)
from optimize.hyperopt_loss import StatArbHyperOptLoss
from optimize.reporting import (
    PerformanceReport,
    ReportingError,
    compute_performance_report,
    print_performance_report,
    render_performance_report,
)
from optimize.walk_forward import (
    WalkForwardError,
    WalkForwardReport,
    WalkForwardRunner,
    WalkForwardWindow,
    WindowResult,
    generate_windows,
)

__all__ = [
    "HyperoptConfig",
    "HyperoptError",
    "HyperoptLauncher",
    "HyperoptResult",
    "Objective",
    "ParameterCombination",
    "ParameterSpec",
    "PerformanceReport",
    "ReportingError",
    "SearchError",
    "SearchResult",
    "StatArbHyperOptLoss",
    "Trial",
    "WalkForwardError",
    "WalkForwardReport",
    "WalkForwardRunner",
    "WalkForwardWindow",
    "WindowResult",
    "build_hyperopt_command",
    "compute_performance_report",
    "generate_windows",
    "grid_search",
    "print_performance_report",
    "random_search",
    "render_performance_report",
]
