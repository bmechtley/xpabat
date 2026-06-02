#!/usr/bin/env bash
#
# Push the locally-migrated split call data to the piku server.
#
# Only the calls/ split JSON is transferred — the server already has the tile
# PNGs in the old <stem>_tiles/ layout, which the new code reads via the
# resolve_*() backwards-compat fallback.  (Run with --tiles to also copy the
# new spectrograms/ tree, e.g. for a clean server with no old tiles.)
#
# Works with macOS's openrsync (no --mkpath / --protect-args): remote dirs are
# pre-created over ssh, and spaces in the <stem> are backslash-escaped so the
# remote shell treats each path as a single argument.
#
# Usage:
#   ./deploy_generated.sh            # calls/ only (recommended)
#   ./deploy_generated.sh --tiles    # calls/ + spectrograms/
#
# After this completes, push the code:  git push piku main
#
set -euo pipefail

REMOTE_HOST="pi.local"
REMOTE_DIR="/home/piku/.piku/apps/bats/generated"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/generated"

WITH_TILES=0
[[ "${1:-}" == "--tiles" ]] && WITH_TILES=1

if ! ping -c 1 -t 3 "$REMOTE_HOST" >/dev/null 2>&1; then
  echo "ERROR: $REMOTE_HOST is unreachable." >&2
  exit 1
fi

echo "Transferring split call data → $REMOTE_HOST:$REMOTE_DIR"
echo

push_dir() {           # $1 = local source dir, $2 = stem, $3 = remote subdir
  local src="$1" stem="$2" sub="$3"
  # The app dir is owned by piku:www-data; the ssh user can't write there.
  # Pre-create with sudo, then transfer via `sudo rsync` on the remote side.
  # Single-quote for the ssh shell; backslash-escape spaces for the rsync shell.
  ssh "$REMOTE_HOST" "sudo mkdir -p '$REMOTE_DIR/$stem/$sub'"
  local stem_esc="${stem// /\\ }"
  rsync -ah --progress --rsync-path="sudo rsync" \
    "$src/" \
    "$REMOTE_HOST:$REMOTE_DIR/$stem_esc/$sub/"
}

for stem_dir in "$LOCAL_DIR"/*/; do
  stem="$(basename "$stem_dir")"
  echo "── $stem ──"
  push_dir "$stem_dir/calls" "$stem" "calls"
  if [[ "$WITH_TILES" == "1" ]]; then
    push_dir "$stem_dir/spectrograms" "$stem" "spectrograms"
  fi
  echo
done

# Hand ownership to the app user so it can write new tiles into the tree.
echo "Setting ownership → piku:www-data"
ssh "$REMOTE_HOST" "sudo chown -R piku:www-data '$REMOTE_DIR'"

echo "Done.  Now deploy the code:  git push piku main"
