#!/usr/bin/env bash
# Publish willitllm to the web server via SFTP (curl + libssh2, no extra deps).
#
# Usage:  bash meta/deploy/deploy.sh
#
# Config: meta/deploy/deploy.cfg  (gitignored)
#         Copy meta/deploy/deploy.cfg.example, fill in credentials, never commit it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CFG="$SCRIPT_DIR/deploy.cfg"

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: $CFG not found."
  echo "       Copy meta/deploy/deploy.cfg.example to meta/deploy/deploy.cfg and fill in credentials."
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
  coder.html
  styles.css
  app.js
  app.calc.js
  app.render.js
  app.ui.js
  coder.js
  data.gpus.js
  data.kv-cache.js
  data.libraries.js
  data.models.js
  data.quantizations.js
)

echo "Publishing to $USER@$HOST:$REMOTE_DIR"
echo

# Fetch the manifest from the server — a plain "hash  filename" file we maintain there.
# If it doesn't exist yet (first deploy), MANIFEST is just empty and everything uploads.
MANIFEST=$(curl --silent --insecure --user "$USER:$PASS" \
  "sftp://$HOST$REMOTE_DIR/manifest.md5" 2>/dev/null || true)

ok=0; skip=0; fail=0
for f in "${FILES[@]}"; do
  local_path="$REPO_ROOT/$f"
  if [[ ! -f "$local_path" ]]; then
    echo "  SKIP  $f  (not found locally)"
    continue
  fi

  # Skip the file if its MD5 matches what we last uploaded.
  local_hash=$(md5sum "$local_path" | cut -d' ' -f1)
  remote_hash=$(echo "$MANIFEST" | awk -v f="$f" '$2==f{print $1}')
  if [[ "$local_hash" == "$remote_hash" ]]; then
    echo "  SKIP  $f  (unchanged)"
    ((skip++)) || true
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

# Only update the manifest when something actually changed and nothing failed,
# so it stays in sync with what's really on the server.
if [[ $ok -gt 0 && $fail -eq 0 ]]; then
  (cd "$REPO_ROOT" && md5sum "${FILES[@]}") | \
    curl --silent --insecure --user "$USER:$PASS" \
         --upload-file - "sftp://$HOST$REMOTE_DIR/manifest.md5"
fi

echo
echo "Done: $ok uploaded, $skip skipped, $fail failed."
[[ $fail -eq 0 ]]
