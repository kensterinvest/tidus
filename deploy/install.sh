#!/usr/bin/env bash
# Tidus VPS install script.
#
# Run ONCE on the z-tidus VPS as root (e.g. via `ssh ionos`):
#   curl -sSL https://raw.githubusercontent.com/kensterinvest/tidus/main/deploy/install.sh | sudo bash
# or locally after `git clone`:
#   sudo bash deploy/install.sh
#
# Idempotent: re-running is safe.

set -euo pipefail

TIDUS_USER=tidus
TIDUS_HOME=/opt/tidus
ENV_DIR=/etc/tidus
ENV_FILE="$ENV_DIR/env"
LOG_DIR=/var/log/tidus
REPO_URL=git@github.com:kensterinvest/tidus.git
SYSTEMD_DIR=/etc/systemd/system

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root." >&2
    exit 1
fi

# 1. System user
if ! id -u "$TIDUS_USER" >/dev/null 2>&1; then
    echo "Creating system user $TIDUS_USER..."
    useradd --system --create-home --home-dir "$TIDUS_HOME" \
            --shell /usr/sbin/nologin "$TIDUS_USER"
fi

# 2. Directories
install -d -o "$TIDUS_USER" -g "$TIDUS_USER" -m 0755 "$TIDUS_HOME"
install -d -o root -g "$TIDUS_USER" -m 0750 "$ENV_DIR"
install -d -o "$TIDUS_USER" -g "$TIDUS_USER" -m 0755 "$LOG_DIR"

# 3. SSH key for git pull/push (deploy key)
if [ ! -f "$TIDUS_HOME/.ssh/id_ed25519" ]; then
    echo "Generating deploy key at $TIDUS_HOME/.ssh/id_ed25519 ..."
    sudo -u "$TIDUS_USER" mkdir -p "$TIDUS_HOME/.ssh"
    chmod 700 "$TIDUS_HOME/.ssh"
    sudo -u "$TIDUS_USER" ssh-keygen -t ed25519 -C "tidus-deploy@z-tidus" \
        -f "$TIDUS_HOME/.ssh/id_ed25519" -N ""
    echo
    echo "=============================================================="
    echo "Add the public key BELOW to:"
    echo "   https://github.com/kensterinvest/tidus/settings/keys"
    echo "with WRITE access enabled."
    echo "=============================================================="
    cat "$TIDUS_HOME/.ssh/id_ed25519.pub"
    echo "=============================================================="
    echo "Press ENTER once you've added the key..."
    read -r _
fi

# 4. Trust GitHub's host key
sudo -u "$TIDUS_USER" ssh-keyscan -t rsa,ecdsa,ed25519 github.com \
    >> "$TIDUS_HOME/.ssh/known_hosts" 2>/dev/null
sudo -u "$TIDUS_USER" sort -u "$TIDUS_HOME/.ssh/known_hosts" \
    -o "$TIDUS_HOME/.ssh/known_hosts"

# 5. Clone (or pull) the repo
if [ ! -d "$TIDUS_HOME/.git" ]; then
    echo "Cloning $REPO_URL ..."
    sudo -u "$TIDUS_USER" git clone "$REPO_URL" "$TIDUS_HOME"
else
    echo "Pulling latest..."
    sudo -u "$TIDUS_USER" git -C "$TIDUS_HOME" fetch origin
    sudo -u "$TIDUS_USER" git -C "$TIDUS_HOME" reset --hard origin/main
fi

# 6. Configure git identity for sync commits
sudo -u "$TIDUS_USER" git -C "$TIDUS_HOME" config user.name  "tidus-bot[bot]"
sudo -u "$TIDUS_USER" git -C "$TIDUS_HOME" config user.email "tidus-bot@z-tidus.com"

# 7. Install uv if missing (the tidus user installs it locally)
if ! sudo -u "$TIDUS_USER" bash -c 'command -v uv >/dev/null 2>&1 || test -x ~/.local/bin/uv'; then
    echo "Installing uv for $TIDUS_USER ..."
    sudo -u "$TIDUS_USER" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi

# Make uv resolvable in subsequent sudo -u calls
TIDUS_PATH="$TIDUS_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

# 8. Create venv + install deps
sudo -u "$TIDUS_USER" env PATH="$TIDUS_PATH" bash -c "cd $TIDUS_HOME && uv venv && uv sync --frozen"

# 9. Env file (placeholder if missing — user must fill in secrets)
if [ ! -f "$ENV_FILE" ]; then
    cat >"$ENV_FILE" <<'EOF'
# Tidus environment — secrets for pricing-sync + magazine delivery.
# This file is owned by root:tidus, mode 0640.
# After editing: `systemctl restart tidus-web && systemctl daemon-reload`.

# Email delivery (Resend; required for magazine)
RESEND_API_KEY=
TIDUS_SMTP_FROM=Tidus Reports <onboarding@resend.dev>

# AI verifier (optional — fails open if absent)
ANTHROPIC_API_KEY=

# Vendor discovery keys (optional — each one missing means that vendor's
# /v1/models endpoint is skipped in discovery)
OPENAI_API_KEY=
GOOGLE_API_KEY=
MISTRAL_API_KEY=
DEEPSEEK_API_KEY=
XAI_API_KEY=

# Pipeline tunables
TIDUS_CANARY_SAMPLE_SIZE=0
EOF
    chown root:"$TIDUS_USER" "$ENV_FILE"
    chmod 0640 "$ENV_FILE"
    echo
    echo "=============================================================="
    echo "Edit $ENV_FILE and fill in real secrets, then re-run this script"
    echo "or just: systemctl restart tidus-web tidus-sync"
    echo "=============================================================="
fi

# 10. Install systemd units (idempotent)
install -m 0644 "$TIDUS_HOME/deploy/tidus-web.service"   "$SYSTEMD_DIR/"
install -m 0644 "$TIDUS_HOME/deploy/tidus-sync.service"  "$SYSTEMD_DIR/"
install -m 0644 "$TIDUS_HOME/deploy/tidus-sync.timer"    "$SYSTEMD_DIR/"
chmod +x "$TIDUS_HOME/deploy/sync_wrapper.sh"

systemctl daemon-reload
systemctl enable --now tidus-web.service
systemctl enable --now tidus-sync.timer

echo
echo "Installed. Status:"
systemctl --no-pager status tidus-web.service | head -8 || true
systemctl --no-pager list-timers tidus-sync.timer | head -4 || true
echo
echo "Add Caddy block from $TIDUS_HOME/deploy/Caddyfile.snippet to /etc/caddy/Caddyfile,"
echo "then: sudo systemctl reload caddy"
