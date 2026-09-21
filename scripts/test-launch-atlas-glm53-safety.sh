#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/glm53-launch-safety.sh
source "$SCRIPT_DIR/lib/glm53-launch-safety.sh"

assert_eq() {
  [[ "$1" == "$2" ]] || {
    echo "expected '$2', got '$1'" >&2
    exit 1
  }
}

assert_rejected() {
  if glm53_safe_serve_tail "$1" >/dev/null; then
    echo "unsafe override was accepted: $1" >&2
    exit 1
  fi
}

assert_eq "$(glm53_safe_serve_tail '')" '--swap-space-gb 0'
assert_eq \
  "$(glm53_safe_serve_tail '--speculative --num-drafts 2')" \
  '--speculative --num-drafts 2 --swap-space-gb 0'
assert_eq \
  "$(glm53_safe_serve_tail '--trace-file logs/run_1.json --ratio=90% --bind node@rack:2')" \
  '--trace-file logs/run_1.json --ratio=90% --bind node@rack:2 --swap-space-gb 0'
# [:blank:] is space AND tab: an operator's copy-paste keeps working.
assert_eq \
  "$(glm53_safe_serve_tail $'--speculative\t--num-drafts 2')" \
  $'--speculative\t--num-drafts 2 --swap-space-gb 0'

assert_rejected '--swap-space-gb 3'
assert_rejected '--swap-space-gb=3'
assert_rejected '--speculative --swap-space-gb 3'
assert_rejected '--speculative; --swap-space-gb 3'
assert_rejected '--speculative # --swap-space-gb 3'
assert_rejected '--speculative $(printf -- --swap-space-gb) 3'
assert_rejected '--speculative "--swap-space-gb" 3'
assert_rejected $'--speculative\n--swap-space-gb 3'
assert_rejected $'--speculative\r\n--swap-space-gb 3'
assert_rejected '--speculative *'
assert_rejected '--trace-file logs\run.json'
assert_rejected '--trace-file ~/run.json'

echo 'GLM launcher safety tests: PASS'
