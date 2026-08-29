#!/usr/bin/env bash
# Web preflight gate for the blog-subdomain work.
# One command that must pass before anything is committed.
set -uo pipefail
ROOT=/workspace/atlas-blog
export HOME=$ROOT/.tmp/home
export TMPDIR=$ROOT/.tmp
export BUN_INSTALL_CACHE_DIR=$ROOT/.tmp/buncache
export ATLAS_RECIPES_ROOT=/workspace/atlas-recipes/recipes
export ATLAS_BASELINES_ROOT=$ROOT/tests/baselines
mkdir -p "$HOME" "$TMPDIR" "$BUN_INSTALL_CACHE_DIR"

fail=0
step() { printf '\n=== %s ===\n' "$1"; }

for app in site blog; do
  [ -d "$ROOT/$app" ] || { echo "skip $app (absent)"; continue; }
  cd "$ROOT/$app" || exit 1

  step "$app: unit tests"
  if [ -n "$(find src/lib -name '*.test.js' -print -quit 2>/dev/null)" ]; then
    bun test src/lib || fail=1
  else
    echo "(no unit tests)"
  fi

  step "$app: build"
  bun x --bun vite build || fail=1
done

step "contrast budget (chevron field over the site ground)"
if [ -f "$ROOT/.contrast-check.mjs" ]; then
  bun "$ROOT/.contrast-check.mjs" || fail=1
else
  echo "(not yet written)"
fi

step "RESULT"
[ "$fail" -eq 0 ] && echo "GATE: PASS" || echo "GATE: FAIL"
exit "$fail"
