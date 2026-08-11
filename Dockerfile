# Railway research/backtest worker image.
#
# This image is deliberately NOT a live-trading deployment: it never
# runs `freqtrade trade`, holds no exchange credentials, and its
# default command (below) does nothing destructive. It exists to run
# `hermes backtest` / `hermes analyze` (see hermes/cli.py) reproducibly
# on Railway, against research-only historical data -- see RAILWAY.md.
#
# Base image matches this project's pinned Python version (pyproject.toml:
# requires-python = ">=3.11"; the committed .venv reports 3.11.15).
FROM python:3.11-slim

WORKDIR /app

# Runs unbuffered so `hermes`/`freqtrade` log lines appear in Railway's
# log stream immediately instead of waiting on Python's stdout buffer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Installs this repo's own packages (hermes/, stat_arb/, optimize/) in
# editable mode via pyproject.toml, exactly as local development does
# (see README.md) -- no separate packaging step to keep in sync.
RUN pip install --no-cache-dir -e .

# Creates the non-root user the process actually runs as; a batch/backtest
# worker has no reason to run as root, and this container never needs to
# bind a privileged port. The image itself stays on root as the container's
# entry user (see ENTRYPOINT below) solely so it can fix ownership of the
# Railway Volume mounted at /app/user_data/data -- which is created empty
# and root-owned -- before dropping to `hermes` to run the real command.
RUN useradd --create-home --uid 1000 hermes && chown -R hermes:hermes /app

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Inert by default: prints Hermes' own CLI help and exits 0. The actual
# research command (e.g. `hermes backtest ...` or `hermes analyze`) is
# supplied as Railway's deploy start command, set explicitly per run --
# see RAILWAY.md. A container that "does nothing but describe itself"
# unless told otherwise is the safest possible default for infrastructure
# that must never become a live-trading dependency.
CMD ["python", "-m", "hermes", "--help"]
