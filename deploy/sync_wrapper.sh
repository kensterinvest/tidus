#!/usr/bin/env bash
# Tidus pricing-sync + magazine wrapper.
#
# Called by tidus-sync.service (systemd oneshot, timer-fired Sun + Wed 02:00 UTC).
#
# Steps:
#   1. flock against concurrent runs and against web-service writes.
#   2. Preserve any subscribers added since the last sync (the web service
#      writes config/subscribers.yaml directly; git reset --hard would
#      otherwise wipe them).
#   3. git fetch + reset --hard origin/main so we run the latest pipeline.
#   4. Restore preserved subscribers.
#   5. Run scripts/weekly_full_sync.py.
#   6. Commit + push tidus.db, reports/, config/models.auto.yaml, and
#      config/subscribers.yaml back to GitHub.

set -euo pipefail

# systemd-spawned environments don't source ~/.bashrc / ~/.profile, so make
# uv resolvable here regardless of how this script was invoked.
export PATH="/opt/tidus/.local/bin:/usr/local/bin:/usr/bin:/bin"

LOCK_FILE=/var/lock/tidus-sync.lock
TIDUS_DIR=/opt/tidus
SUB_FILE_REL=config/subscribers.yaml
SUB_STAGED=/tmp/tidus-subscribers-staged.yaml
LOG_DIR=/var/log/tidus

mkdir -p "$LOG_DIR"

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Another tidus-sync is running; exiting."; exit 0; }

cd "$TIDUS_DIR"

# Pull secrets / SMTP_FROM into env so weekly_full_sync.py + git push see them.
set -a
[ -f /etc/tidus/env ] && . /etc/tidus/env
set +a

# 1. Preserve local subscriber writes
if git status --porcelain "$SUB_FILE_REL" | grep -q .; then
    cp "$SUB_FILE_REL" "$SUB_STAGED"
    HAVE_LOCAL_SUB=1
fi

# 2. Pull latest code
git fetch origin
git reset --hard origin/main

# 3. Restore preserved subscribers (web service is the only mutator and
# add_subscriber() preserves all existing entries before writing, so the
# staged file is a strict superset of the on-main file).
if [ -n "${HAVE_LOCAL_SUB:-}" ]; then
    cp "$SUB_STAGED" "$SUB_FILE_REL"
    rm -f "$SUB_STAGED"
fi

# 4. Sync dependencies (in case pyproject.toml moved)
uv sync --frozen

# 5. Run the magazine pipeline
export TIDUS_CANARY_SAMPLE_SIZE=0
uv run python scripts/weekly_full_sync.py

# 6. Commit + push
git add tidus.db reports/ config/models.auto.yaml "$SUB_FILE_REL" 2>/dev/null || true
if ! git diff --cached --quiet; then
    TODAY=$(date -u +%Y-%m-%d)
    git commit -m "chore(pricing-sync): DB + reports + auto catalog + subscribers ${TODAY}"
    git push origin HEAD:main
else
    echo "No DB/report/auto-catalog/subscriber changes to commit."
fi
