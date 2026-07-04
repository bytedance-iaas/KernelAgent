#!/usr/bin/env bash
# Re-vendor the Jinja2 prompt templates from the upstream triton_kernel_agent
# package into this skill's templates/ directory.
#
# Run this from inside the KernelAgent repo whenever the upstream templates
# change. The skill ships with the vendored copies so it stays usable
# standalone (outside this repo).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="$SCRIPT_DIR/../../triton_kernel_agent/templates"
TARGET="$SCRIPT_DIR/templates"

if [[ ! -d "$UPSTREAM" ]]; then
  echo "error: upstream templates not found at $UPSTREAM" >&2
  echo "This script only works inside the KernelAgent repo." >&2
  exit 1
fi

rsync -a --delete --exclude '__pycache__' --exclude 'README.md' \
  "$UPSTREAM/" "$TARGET/"

echo "Synced templates from $UPSTREAM to $TARGET/"
git -C "$SCRIPT_DIR" status --short templates/ 2>/dev/null || true
