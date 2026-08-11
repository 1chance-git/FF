#!/bin/sh
# Runs as root (the image's default user) purely to fix ownership of the
# Railway Volume mounted at /app/user_data/data, then drops to the
# non-root `hermes` user before exec'ing the real command.
#
# Why this exists: Railway Volumes are created empty and root-owned.
# The image itself runs as `hermes` (uid 1000, see Dockerfile), so
# without this step `freqtrade`/`hermes` can't create directories or
# files inside the mounted volume (PermissionError on first write).
# chown is cheap and idempotent, so this runs unconditionally on every
# start rather than trying to detect "first run".
set -e

mkdir -p /app/user_data/data
chown -R hermes:hermes /app/user_data/data

exec su -s /bin/sh hermes -c 'exec "$0" "$@"' -- "$@"
