#!/usr/bin/env bash
# Publish willitllm to the web server via SFTP (curl + libssh2, no extra deps).
#
# Usage:  bash dev/deploy.sh
#
# Config: dev/deploy.cfg  (gitignored)
#         Copy dev/deploy.cfg.example, fill in credentials, never commit it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG="$SCRIPT_DIR/deploy.cfg"

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: $CFG not found."
  echo "       Copy dev/deploy.cfg.example to dev/deploy.cfg and fill in credentials."
  exit 1
fi

# Source config (HOST, USER, PASS, REMOTE_DIR)
set -a; source "$CFG"; set +a

for var in HOST USER PASS REMOTE_DIR; do
  if [[ -z "${!var:-}" ]]; then
    echo "ERROR: $var is not set in $CFG"
    exit 1
  fi
done

REMOTE_DIR="${REMOTE_DIR%/}"   # strip trailing slash

FILES=(
  index.html
  styles.css
  app.js
  app.calc.js
  app.render.js
  app.ui.js
  data.gpus.js
  data.kv-cache.js
  data.libraries.js
  data.models.js
  data.quantizations.js
)

echo "Publishing to $USER@$HOST:$REMOTE_DIR"
echo

ok=0; fail=0
for f in "${FILES[@]}"; do
  local_path="$REPO_ROOT/$f"
  if [[ ! -f "$local_path" ]]; then
    echo "  SKIP  $f  (not found locally)"
    continue
  fi
  if curl --silent --show-error \
       --insecure \
       --user "$USER:$PASS" \
       --upload-file "$local_path" \
       "sftp://$HOST$REMOTE_DIR/$f"; then
    echo "  OK    $f"
    ((ok++)) || true
  else
    echo "  FAIL  $f"
    ((fail++)) || true
  fi
done

echo
echo "Done: $ok uploaded, $fail failed."
[[ $fail -eq 0 ]]
