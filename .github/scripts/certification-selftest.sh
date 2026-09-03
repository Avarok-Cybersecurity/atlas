#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# The certification pipeline's self-test.
#
# The Rust half of this pipeline (gate/signing.rs, gate/card.rs) has tests that
# run on every PR. The shell and Python half -- which decides who may stamp, who
# may seal, whether a runner can reach untrusted code, and whether a record was
# signed -- had none. Every one of those guards was verified once, by hand, and
# then trusted forever.
#
# EVERY check here has a NEGATIVE CONTROL: the guard is shown failing on input
# it must reject, not merely passing on input it should accept. A guard that has
# only ever been seen to pass is indistinguishable from a guard that cannot
# fail, and the second kind is worse than no guard at all because it reports
# safety.
#
# Runs offline. No network, no GitHub API, no GPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0

ok()   { PASS=$((PASS+1)); printf '  ok   %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  FAIL %s\n' "$1"; }
# want_rc <expected> <label> <cmd...>
want_rc() {
  local want=$1 label=$2; shift 2
  "$@" >"$TMP/out" 2>&1; local got=$?
  if [ "$got" = "$want" ]; then ok "$label"; else
    bad "$label (expected rc=$want, got rc=$got)"; sed 's/^/       /' "$TMP/out" | head -4
  fi
}

echo "== seal coverage =="
printf 'README.md\n' > "$TMP/one.txt"
want_rc 0 "a codeowner of every path covers the diff" \
  sh -c "python3 .github/scripts/seal-coverage.py .github/CODEOWNERS '@tbraun96' < '$TMP/one.txt'"
# CONTROL: a non-owner must NOT cover.
want_rc 1 "control: a non-owner does not cover" \
  sh -c "python3 .github/scripts/seal-coverage.py .github/CODEOWNERS '@nobody-at-all' < '$TMP/one.txt'"
# CONTROL: an empty sealer list must not cover.
want_rc 1 "control: an empty sealer list does not cover" \
  sh -c "python3 .github/scripts/seal-coverage.py .github/CODEOWNERS '' < '$TMP/one.txt'"
# CONTROL: it must FAIL CLOSED on a pattern it cannot match. This inverts
# gate/codeowners.rs, which is documented fail-OPEN; if this ever flips, a
# seal would silently cover paths its author never reviewed.
for pat in '**/x.rs' 'a?b.rs' 'a[b].rs'; do
  printf '%s  @someone\n*  @tbraun96\n' "$pat" > "$TMP/co"
  want_rc 1 "control: refuses unsupported pattern '$pat'" \
    sh -c "python3 .github/scripts/seal-coverage.py '$TMP/co' '@tbraun96' < '$TMP/one.txt'"
done

echo "== the command runner cannot reach untrusted code =="
want_rc 0 "the workflows as they stand are safe" python3 .github/scripts/assert-cmd-runner-safe.py
mkdir -p "$TMP/wf/.github/workflows"; cp .github/scripts/assert-cmd-runner-safe.py "$TMP/wf/"
# CONTROL x3: each shape that would let a fork run code on our hardware.
cat > "$TMP/wf/.github/workflows/a.yml" <<'Y'
on: { pull_request: { types: [opened] } }
jobs: { j: { runs-on: atlas-cmd, steps: [{ uses: actions/checkout@v4, with: { ref: main } }] } }
Y
want_rc 1 "control: pull_request trigger on the command runner" \
  sh -c "cd '$TMP/wf' && python3 assert-cmd-runner-safe.py"
cat > "$TMP/wf/.github/workflows/a.yml" <<'Y'
on: { pull_request_target: { types: [opened] } }
jobs: { j: { runs-on: [self-hosted, atlas-cmd], steps: [{ uses: actions/checkout@v4, with: { ref: "${{ github.event.pull_request.head.sha }}" } }] } }
Y
want_rc 1 "control: checks out the PR head on the command runner" \
  sh -c "cd '$TMP/wf' && python3 assert-cmd-runner-safe.py"
cat > "$TMP/wf/.github/workflows/a.yml" <<'Y'
on: { issue_comment: { types: [created] } }
jobs: { j: { runs-on: atlas-cmd, steps: [{ uses: actions/checkout@v4 }] } }
Y
want_rc 1 "control: checkout with no explicit ref on the command runner" \
  sh -c "cd '$TMP/wf' && python3 assert-cmd-runner-safe.py"

echo "== certificate rendering =="
T=docs/diagrams/states/certificate-merged.svg
render() { python3 .github/scripts/render-certificate.py --template "$T" --out "$1" \
    --url https://github.com/o/r/pull/7 --pr 7 --title t --repo o/r --commit abc1234567 \
    --date 2026-01-01 --gates "11 / 11" --qr-x 980 --qr-y 455 "${@:2}"; }
want_rc 0 "renders with one author"           render "$TMP/c1.svg" --authors "alice"
want_rc 0 "renders with three authors"        render "$TMP/c3.svg" --authors "alice,bob,carol"
want_rc 0 "renders with the count absent"     render "$TMP/c0.svg" --authors "alice"
# A filled author slot must be VISIBLE. The template ships slots 2 and 3 hidden,
# and an earlier version set their text without unhiding them -- co-authors were
# silently dropped from a certificate that looked correct.
vis() { python3 - "$1" <<'PY'
import re,sys
s=open(sys.argv[1],encoding='utf-8').read()
print(sum(1 for i in (1,2,3)
          if re.search(r'<g[^>]*id="field-cert-author-%d"(?![^>]*display="none")'%i, s)))
PY
}
[ "$(vis "$TMP/c1.svg")" = "1" ] && ok "one author -> one visible slot" || bad "one author -> $(vis "$TMP/c1.svg") visible slots"
[ "$(vis "$TMP/c3.svg")" = "3" ] && ok "three authors -> three visible slots" || bad "three authors -> $(vis "$TMP/c3.svg") visible slots"
# CONTROL: a template that lacks an id must be an error, not a silent no-op.
sed 's/id="value-cert-pr"/id="value-cert-pr-RENAMED"/' "$T" > "$TMP/broken.svg"
want_rc 1 "control: a renamed id is an error, not a silent skip" \
  sh -c "python3 .github/scripts/render-certificate.py --template '$TMP/broken.svg' --out '$TMP/x.svg' --url u --pr 7 --title t --authors alice"
# The rendered SVG must be well-formed; a duplicated attribute made librsvg
# refuse the whole file while the script still exited 0.
want_rc 0 "output parses as XML" python3 -c "import xml.dom.minidom,sys;xml.dom.minidom.parse('$TMP/c3.svg')"

echo "== pr-review survives a large payload =="
# `set -euo pipefail` plus `jq | head -c` takes SIGPIPE past the 64K pipe
# buffer and kills the script. Any PR body over 4000 chars hit it.
python3 -c "import json;print(json.dumps({'body':'x'*300000}))" > "$TMP/big.json"
want_rc 0 "truncation does not die on a 300KB body" \
  bash -c "set -euo pipefail; b=\$(jq -r '(.body // \"\")[0:4000]' '$TMP/big.json'); [ \${#b} -eq 4000 ]"
# CONTROL: the old form must still be shown to fail, or this test proves nothing.
want_rc 141 "control: the pre-fix form still dies on SIGPIPE" \
  bash -c "set -euo pipefail; b=\$(jq -r '.body // \"\"' '$TMP/big.json' | head -c 4000); echo ok"

echo
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
