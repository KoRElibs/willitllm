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

# ── File list — built from glob patterns at deploy time ──────────────────────
# Each pattern is expanded against REPO_ROOT. Unknown files are ignored safely
# because we use nullglob. Add patterns here when introducing new file groups.

INCLUDE_PATTERNS=(
  "*.html"
  "*.css"
  "app.*.js"      # app.calc.js  app.fmt.js  app.shared.js  app.util.js
  "index.js"      # main entry point (no dot-segment)
  "index.*.js"    # index.combobox.js  index.variants.js  index.ui.js …
  "coder.js"      # main entry point (no dot-segment)
  "coder.*.js"    # coder.rank.js  coder.rows.js
  "data.*.js"     # all data files
)

FILES=()
cd "$REPO_ROOT"
shopt -s nullglob
for pat in "${INCLUDE_PATTERNS[@]}"; do
  for f in $pat; do
    [[ -f "$f" ]] && FILES+=("$f")
  done
done
shopt -u nullglob

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "ERROR: No files matched the include patterns. Check REPO_ROOT=$REPO_ROOT"
  exit 1
fi

# ── SFTP helpers ──────────────────────────────────────────────────────────────

# Ensure the SFTP host is in known_hosts — curl with libssh (Ubuntu 22.04+) silently
# fails with error 78 if the host is unknown rather than giving a clear auth error.
if ! ssh-keygen -F "$HOST" &>/dev/null; then
  echo "Adding $HOST to ~/.ssh/known_hosts..."
  FETCHED=$(ssh-keyscan "$HOST" 2>/dev/null)
  if [[ -z "$FETCHED" ]]; then
    echo "ERROR: Could not reach $HOST to fetch its host key. Check your network."
    exit 1
  fi
  echo "$FETCHED" >> ~/.ssh/known_hosts
fi

SFTP_BASE="sftp://$HOST/$REMOTE_DIR"   # double-slash = absolute path (required by libssh)

sftp_upload() {
  local local_path="$1" remote_name="$2"
  curl --silent --show-error --insecure --user "$USER:$PASS" \
    --upload-file "$local_path" "$SFTP_BASE/$remote_name"
}

sftp_delete() {
  # Send 'rm' as a pre-transfer quote on a directory listing (no file upload needed).
  # The directory listing is the nominal "transfer" that establishes the connection.
  curl --silent --show-error --insecure --user "$USER:$PASS" \
    -Q "rm $REMOTE_DIR/$1" \
    "$SFTP_BASE/"
}

# ── Upload changed files ──────────────────────────────────────────────────────

echo "Publishing to $USER@$HOST:$REMOTE_DIR"
echo "Files matched by patterns: ${#FILES[@]}"
echo

# Fetch the manifest from the server — a plain "hash  filename" file we maintain there.
# If it doesn't exist yet (first deploy), MANIFEST is just empty and everything uploads.
MANIFEST=$(curl --silent --insecure --user "$USER:$PASS" \
  "$SFTP_BASE/manifest.md5" 2>/dev/null || true)

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

  if sftp_upload "$local_path" "$f"; then
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
         --upload-file - "$SFTP_BASE/manifest.md5"
fi

echo
echo "Done: $ok uploaded, $skip skipped, $fail failed."

# ── Stale server-side file cleanup ───────────────────────────────────────────
# Files in the manifest (= previously deployed) but absent from the current
# pattern-matched list are relics of old deploys. Offer to remove them.

if [[ -n "$MANIFEST" ]]; then
  # Build a lookup of current files
  declare -A current_set
  for f in "${FILES[@]}"; do
    current_set[$f]=1
  done

  stale=()
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    fname=$(awk '{print $2}' <<< "$line")
    [[ -z "$fname" || "$fname" == "manifest.md5" ]] && continue
    [[ -z "${current_set[$fname]:-}" ]] && stale+=("$fname")
  done <<< "$MANIFEST"

  if [[ ${#stale[@]} -gt 0 ]]; then
    echo
    echo "Server has ${#stale[@]} file(s) no longer in the deploy list:"
    for f in "${stale[@]}"; do
      echo "    $f"
    done
    echo
    read -rp "Delete these from the server? [y/N] " answer
    if [[ "${answer,,}" == "y" ]]; then
      del_ok=0; del_fail=0
      for f in "${stale[@]}"; do
        if sftp_delete "$f"; then
          echo "  DEL   $f"
          ((del_ok++)) || true
        else
          echo "  FAIL  DEL $f"
          ((del_fail++)) || true
        fi
      done
      echo "Deleted: $del_ok ok, $del_fail failed."

      # Rebuild manifest without the deleted files
      if [[ $del_fail -eq 0 && $fail -eq 0 ]]; then
        (cd "$REPO_ROOT" && md5sum "${FILES[@]}") | \
          curl --silent --insecure --user "$USER:$PASS" \
               --upload-file - "$SFTP_BASE/manifest.md5"
      fi
    else
      echo "Skipped. Run again to be prompted next time (stale files stay in manifest)."
    fi
  fi
fi

[[ $fail -eq 0 ]]
