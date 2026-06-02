#!/usr/bin/env bash
#
# Push the locally-migrated split call data to the piku server.
#
# Only the calls/ split JSON is transferred — the server already has the tile
# PNGs in the old <stem>_tiles/ layout, which the new code reads via the
# resolve_*() backwards-compat fallback.  (Run with --tiles to also copy the
# new spectrograms/ tree, e.g. for a clean server with no old tiles.)
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

for stem_dir in "$LOCAL_DIR"/*/; do
  stem="$(basename "$stem_dir")"
  echo "── $stem ──"

  # Always sync calls/ (the split JSON)
  rsync -avh --progress \
    "$stem_dir/calls/" \
    "$REMOTE_HOST:$REMOTE_DIR/$stem/calls/"

  if [[ "$WITH_TILES" == "1" ]]; then
    rsync -avh --progress \
      "$stem_dir/spectrograms/" \
      "$REMOTE_HOST:$REMOTE_DIR/$stem/spectrograms/"
  fi
  echo
done

echo "Done.  Now deploy the code:  git push piku main"
