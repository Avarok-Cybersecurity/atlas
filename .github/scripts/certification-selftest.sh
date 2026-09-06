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
TMP=$(mktemp -d)

# A suite that exits early is indistinguishable from a suite that ran and
# failed: same non-zero status, no summary line, and every check after the exit
# point silently not run. That shipped here -- a stray `set -e` left over from
# one control turned errexit on, and the next control that was SUPPOSED to fail
# killed the run after 97 checks, reporting nothing but exit 1.
REACHED_SUMMARY=0
trap 'rm -rf "$TMP"; [ "$REACHED_SUMMARY" = 1 ] || { echo; echo "  SUITE TRUNCATED: exited after $PASS checks, before the summary."; echo "  Everything after that point did not run. This is not a normal failure."; }' EXIT
PASS=0; FAIL=0

# Prerequisites, checked up front and LOUDLY. The alternative -- skipping the
# checks whose tools are missing -- turns a suite that cannot run into a suite
# that reports success, which is the exact failure this file exists to prevent.
missing=""
for tool in python3 jq; do command -v "$tool" >/dev/null || missing="$missing $tool"; done
python3 -c 'import yaml' 2>/dev/null || missing="$missing python3-yaml"
python3 -c 'import segno' 2>/dev/null || missing="$missing python3-segno"
if [ -n "$missing" ]; then
  echo "cannot run: missing$missing" >&2
  echo "install them and re-run; this suite never skips a check it cannot perform." >&2
  exit 2
fi

ok()   { PASS=$((PASS+1)); printf '  ok   %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  FAIL %s\n' "$1"; }

# A control that calls a helper this file does not define runs as NOTHING.
# bash prints "command not found" on stderr, the assertion never executes, and
# the suite's totals do not move -- so deleting six controls reads as
# "134 passed, 0 failed" instead of six failures. That happened here on
# 2026-09-06: `want_out` was used by rows ported from #876 but its DEFINITION
# stayed behind, and both a local run and a CI run reported a clean pass.
#
# This turns the silence into a failure. Anything bash cannot resolve -- a
# missing helper, a typo'd assertion name, a tool absent from the runner --
# now costs a FAIL that names the word it could not find.
command_not_found_handle() {
  bad "INTERNAL: '$1' is not a command or function — whatever control invoked it did nothing"
  return 127
}
# want_broken_pipe <label> <cmd...> -- asserts the command FAILS because of a
# broken pipe, without pinning the exit code. jq traps EPIPE and exits 2 with a
# message; tools that take the signal die with 141. Both are the failure this
# control exists to demonstrate, and which one you get depends on the jq build,
# so asserting 141 made the control pass on one machine and fail on another.
want_nonzero() {
  local label=$1; shift
  "$@" >"$TMP/out" 2>&1; local got=$?
  if [ "$got" -ne 0 ]; then ok "$label"; else
    bad "$label (expected a non-zero exit, got 0)"; sed 's/^/       /' "$TMP/out" | head -3
  fi
}

# want_rc_msg <expected-rc> <substring> <label> <cmd...> -- rc AND which guard
# fired. Several guards here overlap: a record with no sidecar trips the
# signature check AND the one-signer check, because "no sidecar" reads as a
# distinct signer. A control that only asserts rc=1 therefore passes even when
# the guard it targets has been deleted -- verified: removing the signature
# check left the suite green at 32/32.
want_rc_msg() {
  local want=$1 needle=$2 label=$3; shift 3
  "$@" >"$TMP/out" 2>&1; local got=$?
  if [ "$got" = "$want" ] && grep -qF "$needle" "$TMP/out"; then ok "$label"; else
    bad "$label (rc=$got want=$want; expected message containing '$needle')"
    sed 's/^/       /' "$TMP/out" | head -4
  fi
}

# want_out <expected-stdout> <label> <cmd...> -- for a guard whose verdict is a
# VALUE it prints, not an exit code. `stack_says` returns a count, and a count
# of 0 and a count of 2 are both a clean exit.
#
# ★ This function was MISSING when the stack rows were ported here from #876,
# and bash's answer to an undefined function is "command not found" on stderr
# and a non-zero status that nothing was reading. Six controls therefore ran as
# NOTHING -- no ok, no FAIL, no effect on the totals -- while the suite printed
# a clean "133 passed, 0 failed" both locally and on the runner. A control that
# cannot be seen to fail is not a control, and one that cannot be seen to RUN
# is not even a line of code.
want_out() {
  local want=$1 label=$2; shift 2
  local got; got=$("$@" 2>"$TMP/err")
  if [ "$got" = "$want" ]; then ok "$label"; else
    bad "$label (printed '$got', want '$want')"; sed 's/^/       /' "$TMP/err" | head -3
  fi
}

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

# The cheap PR pool runs PR code on purpose, so its whole safety property is the
# same-repo comparison in `runs-on`. These three controls are that property.
cat > "$TMP/wf/.github/workflows/a.yml" <<'Y'
on: { pull_request: { types: [opened] } }
jobs: { j: { runs-on: atlas-pr-cheap, steps: [{ uses: actions/checkout@v4 }] } }
Y
want_rc 1 "control: cheap pool on pull_request with no same-repo guard" \
  sh -c "cd '$TMP/wf' && python3 assert-cmd-runner-safe.py"
cat > "$TMP/wf/.github/workflows/a.yml" <<'Y'
on: { pull_request: { types: [opened] } }
jobs: { j: { runs-on: "${{ vars.PR_CHEAP_RUNNER || 'ubuntu-latest' }}", steps: [{ uses: actions/checkout@v4 }] } }
Y
want_rc 1 "control: cheap pool via the variable, still unguarded" \
  sh -c "cd '$TMP/wf' && python3 assert-cmd-runner-safe.py"
# Guarded, but then hands the runner a fork ref explicitly -- the guard chooses
# the RUNNER, it does not choose what gets checked out.
cat > "$TMP/wf/.github/workflows/a.yml" <<'Y'
on: { pull_request: { types: [opened] } }
jobs:
  j:
    runs-on: "${{ github.event.pull_request.head.repo.full_name == github.repository && 'atlas-pr-cheap' || 'ubuntu-latest' }}"
    steps: [{ uses: actions/checkout@v4, with: { ref: "${{ github.event.pull_request.head.sha }}" } }]
Y
want_rc 1 "control: cheap pool guarded but checking out a fork ref" \
  sh -c "cd '$TMP/wf' && python3 assert-cmd-runner-safe.py"
# And the shape we actually ship must PASS, or the rule is unusable.
cat > "$TMP/wf/.github/workflows/a.yml" <<'Y'
on: { pull_request: { types: [opened] } }
jobs:
  j:
    runs-on: "${{ (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository) && (vars.PR_CHEAP_RUNNER || 'ubuntu-latest') || 'ubuntu-latest' }}"
    steps: [{ uses: actions/checkout@v4 }]
Y
want_rc 0 "the shipped cheap-pool routing shape is accepted" \
  sh -c "cd '$TMP/wf' && python3 assert-cmd-runner-safe.py"

echo "== the suite cannot silently skip a control =="
# CONTROL: a helper this file does not define must COST a failure, not vanish.
# Run in a subshell so the probe's FAIL does not enter the real totals; assert
# on what the handler printed.
probe=$( (command_not_found_handle no_such_assertion_helper) 2>&1 )
case "$probe" in
  *"INTERNAL: 'no_such_assertion_helper' is not a command or function"*)
    ok "an undefined helper is reported, not silently skipped" ;;
  *) bad "an undefined helper vanished silently: $probe" ;;
esac

echo "== a reusable workflow is called with the inputs it declares =="
G=.github/scripts/assert-reusable-workflow-inputs.py
want_rc 0 "the workflows as they stand match their callees" python3 "$G"
mkdir -p "$TMP/ri/.github/workflows"
mk_callee() { cat > "$TMP/ri/.github/workflows/callee.yml" <<Y
on:
  workflow_call:
    inputs:
      web_only: { required: false, type: boolean, default: false }
      $1
jobs: { j: { runs-on: ubuntu-latest, steps: [{ run: "true" }] } }
Y
}
mk_caller() { cat > "$TMP/ri/.github/workflows/caller.yml" <<Y
on: { pull_request: { types: [opened] } }
jobs:
  call:
    uses: ./.github/workflows/callee.yml
    with:
$1
Y
}
# The shape that must PASS: every key passed is declared.
mk_callee 'flag: { required: false, type: boolean, default: false }'
mk_caller '      web_only: true
      flag: true'
want_rc 0 "a call site passing only declared inputs is accepted" python3 "$G" "$TMP/ri"
# CONTROL: the 2026-09-06 near-miss -- `with:` carries a key whose DECLARATION
# was left behind in an unported commit. GitHub rejects the whole calling
# workflow at dispatch, so every context it would report is never created and
# branch protection waits forever on a check nobody will write.
mk_callee 'flag: { required: false, type: boolean, default: false }'
mk_caller '      web_only: true
      stack_layer: true'
want_rc 1 "control: an input passed but never declared is refused" python3 "$G" "$TMP/ri"
# CONTROL: the same dispatch failure from the other side.
mk_callee 'must_have: { required: true, type: string }'
mk_caller '      web_only: true'
want_rc 1 "control: a required input the caller omits is refused" python3 "$G" "$TMP/ri"
# NOT flagged: `workflow_call:` with nothing under it is a valid callee taking
# no inputs. The first draft tested the VALUE rather than the KEY and reported
# ci.yml as "not a workflow_call workflow" -- a false alarm about release.yml.
cat > "$TMP/ri/.github/workflows/callee.yml" <<'Y'
on:
  workflow_call:
jobs: { j: { runs-on: ubuntu-latest, steps: [{ run: "true" }] } }
Y
cat > "$TMP/ri/.github/workflows/caller.yml" <<'Y'
on: { pull_request: { types: [opened] } }
jobs: { call: { uses: ./.github/workflows/callee.yml } }
Y
want_rc 0 "an input-less workflow_call callee is not mistaken for a non-callee" \
  python3 "$G" "$TMP/ri"

echo "== a required check cannot be cancelled by comment timing =="
want_rc 0 "the workflows as they stand are safe" \
  python3 .github/scripts/assert-required-checks-not-comment-cancellable.py
mkdir -p "$TMP/cc/.github/workflows"
# CONTROL: the exact shape that blocked #907 -- required context + comment
# trigger + cancel-in-progress.
cat > "$TMP/cc/.github/workflows/a.yml" <<'Y'
on: { issue_comment: { types: [created] } }
concurrency: { group: g, cancel-in-progress: true }
jobs: { CLAAssistant: { name: CLAAssistant, runs-on: ubuntu-latest, steps: [{ run: "true" }] } }
Y
want_rc 1 "control: required check cancellable by a comment" \
  python3 .github/scripts/assert-required-checks-not-comment-cancellable.py "$TMP/cc"
# A job-level concurrency block must be seen too, not just the workflow one.
cat > "$TMP/cc/.github/workflows/a.yml" <<'Y'
on: { issue_comment: { types: [created] } }
jobs:
  CLAAssistant:
    name: CLAAssistant
    runs-on: ubuntu-latest
    concurrency: { group: g, cancel-in-progress: true }
    steps: [{ run: "true" }]
Y
want_rc 1 "control: job-level concurrency is checked as well" \
  python3 .github/scripts/assert-required-checks-not-comment-cancellable.py "$TMP/cc"
# NOT flagged: a push-triggered required check. There a cancellation is a
# supersession and the replacement run reports, so forbidding it would be wrong.
cat > "$TMP/cc/.github/workflows/a.yml" <<'Y'
on: { pull_request: { types: [opened] } }
concurrency: { group: g, cancel-in-progress: true }
jobs: { typos: { name: typos, runs-on: ubuntu-latest, steps: [{ run: "true" }] } }
Y
want_rc 0 "a push-triggered required check may still cancel itself" \
  python3 .github/scripts/assert-required-checks-not-comment-cancellable.py "$TMP/cc"
# NOT flagged: a comment-triggered job that is not a required context.
cat > "$TMP/cc/.github/workflows/a.yml" <<'Y'
on: { issue_comment: { types: [created] } }
concurrency: { group: g, cancel-in-progress: true }
jobs: { advisory: { name: "PR categorize (advisory)", runs-on: ubuntu-latest, steps: [{ run: "true" }] } }
Y
want_rc 0 "a non-required comment-triggered job may cancel itself" \
  python3 .github/scripts/assert-required-checks-not-comment-cancellable.py "$TMP/cc"

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

# THE SPECIMEN MUST NEVER SURVIVE. The template is artwork drawn on a specimen
# PR -- its title, its commit and its author are all real-looking strings sitting
# in the file. A row with no group cannot be hidden, so an empty value used to be
# answered by SKIPPING the substitution, which does not blank the row: it leaves
# the specimen's data standing and publishes it as this PR's. Reachable, not
# theoretical -- the bot reads both the title and the merge sha through
# `gh api ... 2>/dev/null`, so either comes back empty on a rate limit or a 404.
SPECIMEN_TITLE='Fuse the GDN spine epilogue into the decode kernel'
SPECIMEN_COMMIT='9d4e1f07c2'
SPECIMEN_AUTHOR=$(grep -o 'id="value-cert-author-1"[^>]*>[^<]*' "$T" | sed 's/.*>//')
if grep -qF "$SPECIMEN_TITLE" "$T" && grep -qF "$SPECIMEN_COMMIT" "$T" && [ -n "$SPECIMEN_AUTHOR" ]; then
  ok "the template does carry specimen data (so the checks below mean something)"
else
  bad "setup: the template no longer carries the specimen strings these checks look for"
fi

want_nonzero "an empty --title is refused, not drawn over with the specimen's" \
  render "$TMP/spec-t.svg" --authors alice --title ""
want_nonzero "an empty --commit is refused, not drawn over with the specimen's" \
  render "$TMP/spec-c.svg" --authors alice --commit ""
render "$TMP/spec-ok.svg" --authors alice >/dev/null 2>&1
if grep -qF "$SPECIMEN_TITLE" "$TMP/spec-ok.svg" || grep -qF "$SPECIMEN_COMMIT" "$TMP/spec-ok.svg"; then
  bad "a rendered certificate still carries the template's specimen data"
else
  ok "a rendered certificate carries none of the template's specimen data"
fi

# CONTROL: the pre-fix behaviour -- skip the substitution when the value is
# empty -- restored in a COPY. It must publish the specimen's title, which is
# what proves the three checks above are not passing by construction.
mkdir -p "$TMP/rc"
python3 - .github/scripts/render-certificate.py "$TMP/rc/render.py" <<'RCSAB'
import pathlib, re, sys
t = pathlib.Path(sys.argv[1]).read_text()
new, n = re.subn(r'        if not str\(val\)\.strip\(\):\n(?:            .*\n)+',
                 '        if not str(val).strip():\n            continue\n', t)
assert n == 1, "sabotage would not land: the empty-value guard was not found"
pathlib.Path(sys.argv[2]).write_text(new)
RCSAB
python3 "$TMP/rc/render.py" --template "$T" --out "$TMP/rc/out.svg" \
  --url https://github.com/o/r/pull/7 --pr 7 --title "" --repo o/r --commit abc1234567 \
  --date 2026-01-01 --gates "11 / 11" --authors alice --qr-x 980 --qr-y 455 >/dev/null 2>&1
grep -qF "$SPECIMEN_TITLE" "$TMP/rc/out.svg" \
  && ok "control: skipping an empty field publishes the specimen's title" \
  || bad "control: the sabotage did not reproduce the defect -- the checks above prove nothing"

# ZERO authors is reachable too: `who` is built from two `gh api ... 2>/dev/null`
# calls and can come back empty. Slot 1 is the only one the template ships
# VISIBLE, so vis() above cannot see this -- it counts slots the template had
# already hidden, and a run with hide() deleted left every check in this section
# green while a certificate named the specimen author.
render "$TMP/c-none.svg" --authors "" >/dev/null 2>&1
[ "$(vis "$TMP/c-none.svg")" = "0" ] \
  && ok "no authors -> no visible slot (the specimen author is not left standing)" \
  || bad "no authors -> $(vis "$TMP/c-none.svg") visible slots, drawing the specimen's name"
python3 - .github/scripts/render-certificate.py "$TMP/rc/nohide.py" <<'HDSAB'
import pathlib, sys
t = pathlib.Path(sys.argv[1]).read_text()
old = '        else:\n            svg = hide(svg, f"field-cert-author-{i}")\n'
assert t.count(old) == 1, "sabotage would not land: the author hide() was not found"
pathlib.Path(sys.argv[2]).write_text(t.replace(old, '        else:\n            pass\n'))
HDSAB
python3 "$TMP/rc/nohide.py" --template "$T" --out "$TMP/rc/nohide.svg" \
  --url https://github.com/o/r/pull/7 --pr 7 --title t --repo o/r --commit abc1234567 \
  --date 2026-01-01 --gates "11 / 11" --authors "" --qr-x 980 --qr-y 455 >/dev/null 2>&1
[ "$(vis "$TMP/rc/nohide.svg")" != "0" ] \
  && ok "control: dropping hide() leaves the specimen author visible" \
  || bad "control: the sabotage did not reproduce the defect -- the check above proves nothing"
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
# The exit code differs by platform -- jq traps EPIPE and exits 2 with a
# message, other builds take the signal and die with 141 silently -- so pinning
# either one made this control pass on one machine and fail on another. What is
# invariant, and what the control actually needs to show, is that the PIPED form
# fails on input where the UNPIPED form above succeeded. The pipe is the cause.
want_nonzero "control: the pre-fix (piped) form fails on the same input" \
  bash -c "set -euo pipefail; b=\$(jq -r '.body // \"\"' '$TMP/big.json' | head -c 4000); echo ok"

echo "== ci.yml decision logic (extracted and run against a stubbed API) =="
# These shell blocks decide whether anything merges uncertified. They live
# inside ci.yml where nothing could execute them, so they are pulled out and run
# here against a fake `gh` -- no network, deterministic.
extract() { python3 -c '
import sys, yaml
d = yaml.safe_load(open(".github/workflows/ci.yml"))
job, name = sys.argv[1].split("/", 1)
for st in d["jobs"][job]["steps"]:
    if (st.get("name") or "") == name or (name == "-" and st.get("run")):
        print(st["run"]); break
' "$1"; }

mkstub() {
  mkdir -p "$TMP/bin"
  printf '#!/bin/bash\ncase "$*" in\n  *check-runs*) echo "%s" ;;\n  *pulls/*/commits*) echo "" ;;\n  *pulls/*) echo deadbeefcafe ;;\n  *) echo "" ;;\nesac\n' "$1" > "$TMP/bin/gh"
  chmod +x "$TMP/bin/gh"
}

extract 'stamp/-' > "$TMP/stamp.sh"
if [ -s "$TMP/stamp.sh" ]; then
  mkstub 1
  : > "$TMP/go1"
  ( PATH="$TMP/bin:$PATH" GITHUB_OUTPUT="$TMP/go1" REPO=o/r EVENT=pull_request \
    SHA=abc PR=1 MQ_HEAD_REF= bash "$TMP/stamp.sh" >/dev/null 2>&1 )
  grep -q 'stamped=true' "$TMP/go1" && ok "stamp: a Stamp releases the lane" || bad "stamp: a Stamp did not release the lane"
  mkstub 0
  : > "$TMP/go2"
  ( PATH="$TMP/bin:$PATH" GITHUB_OUTPUT="$TMP/go2" REPO=o/r EVENT=pull_request \
    SHA=abc PR=1 MQ_HEAD_REF= bash "$TMP/stamp.sh" >/dev/null 2>&1 )
  grep -q 'stamped=false' "$TMP/go2" && ok "control: no Stamp holds the lane" || bad "control: an unstamped PR did not hold the lane"
  : > "$TMP/go3"
  ( PATH="$TMP/bin:$PATH" GITHUB_OUTPUT="$TMP/go3" REPO=o/r EVENT=merge_group \
    SHA=abc PR= MQ_HEAD_REF= bash "$TMP/stamp.sh" >/dev/null 2>&1 )
  grep -q 'stamped=true' "$TMP/go3" && ok "merge_group is never held" || bad "merge_group was held -- that would wedge the queue"
else
  bad "could not extract the stamp shell from ci.yml"
fi

extract 'pr-benchmark-gate-alias/Mirror the certification verdict' > "$TMP/alias.sh"
if [ -s "$TMP/alias.sh" ]; then
  want_rc 0 "alias: a green certification passes" \
    env RESULT=success WEB_ONLY=false STAMPED=true EXPEDITED=false bash "$TMP/alias.sh"
  want_rc 0 "alias: a web-only diff passes on a deliberate skip" \
    env RESULT=skipped WEB_ONLY=true STAMPED=true EXPEDITED=false bash "$TMP/alias.sh"
  want_rc 0 "alias: an admin expedite passes" \
    env RESULT=skipped WEB_ONLY=false STAMPED=true EXPEDITED=true bash "$TMP/alias.sh"
  want_rc 1 "control: held (unstamped) stays red" \
    env RESULT=skipped WEB_ONLY=false STAMPED=false EXPEDITED=false bash "$TMP/alias.sh"
  want_rc 1 "control: a failed certification stays red" \
    env RESULT=failure WEB_ONLY=false STAMPED=true EXPEDITED=false bash "$TMP/alias.sh"
  # The one that matters most: a BROKEN classifier leaves WEB_ONLY empty. If that
  # ever passed, any diff could skip certification by breaking classify-diff.
  want_rc 1 "control: a broken classifier (empty web_only) stays red" \
    env RESULT=skipped WEB_ONLY= STAMPED=true EXPEDITED=false bash "$TMP/alias.sh"
else
  bad "could not extract the alias shell from ci.yml"
fi

# ── The classify side: what DECIDES is_stack_layer ──────────────────────────
# The alias rows above prove the verdict is translated correctly. These prove
# the verdict itself is reached correctly -- including that every way the
# lookup can go wrong lands on "certify".
extract 'changes/Stack position' > "$TMP/stack.sh"
if [ -s "$TMP/stack.sh" ]; then
  # `gh pr list ... --jq length` prints the number of open PRs stacked on top.
  stub_gh_count() { printf '#!/bin/bash\nprintf "%s\\n"\n' "$1" > "$TMP/bin/gh"; chmod +x "$TMP/bin/gh"; }
  # stack_says <event> <head_repo> -- head ref fixed; the layer test must not
  # depend on the base ref at all (see the workflow comment: the top of a
  # native stack has a non-main base and is the PR that lands on main).
  stack_says() {
    mkdir -p "$TMP/bin"; : > "$TMP/gh_out"
    env PATH="$TMP/bin:$PATH" GITHUB_OUTPUT="$TMP/gh_out" \
        GITHUB_EVENT_NAME="$1" HEAD_REPO="$2" HEAD_REF=feat/mine REPO=o/r \
        bash "$TMP/stack.sh" >/dev/null 2>&1
    grep -c "^is_stack_layer=true$" "$TMP/gh_out"
  }

  stub_gh_count 0
  want_out 0 "stack: a PR with nothing stacked above it certifies — whatever its base" stack_says pull_request o/r
  stub_gh_count 2
  want_out 1 "stack: a PR with 2 PRs stacked above it is a lower layer" stack_says pull_request o/r

  # FAIL-SAFE, every way the classification can break. Each must CERTIFY
  # (false), never skip: an extra campaign is recoverable, an uncertified
  # merge is not.
  printf '#!/bin/bash\nexit 1\n' > "$TMP/bin/gh"; chmod +x "$TMP/bin/gh"
  want_out 0 "control: a failing gh call certifies rather than skipping" stack_says pull_request o/r
  stub_gh_count "not-a-number"
  want_out 0 "control: garbage from gh certifies rather than skipping" stack_says pull_request o/r
  stub_gh_count 2
  want_out 0 "control: a merge_group run certifies (no PR context to read)" stack_says merge_group o/r
  # A FORK PR's head branch name can coincide with a stack's base branch in
  # this repo -- `--base` matches by name. If that coincidence ever counted as
  # "stacked above", any fork could skip certification by naming its branch.
  stub_gh_count 2
  want_out 0 "control: a fork PR certifies even when its branch name matches a stack base" stack_says pull_request someone/fork
else
  bad "could not extract the stack-position shell from ci.yml"
fi

# ── release matrix: the summary's skip-acceptance logic ─────────────────────
# `dry-run summary` is a required context; its first step converts upstream
# results into a verdict. Exactly two skips are acceptable, and each must be
# NAMED by its input -- a bare skip is still a failure.
extract_rb() { python3 -c '
import sys, yaml
d = yaml.safe_load(open(".github/workflows/release-build.yml"))
for st in d["jobs"]["dry-run-summary"]["steps"]:
    if (st.get("name") or "") == sys.argv[1]:
        print(st["run"]); break
' "$1"; }
extract_rb 'Fail explicitly if anything upstream failed' > "$TMP/rbsum.sh"
if [ -s "$TMP/rbsum.sh" ]; then
  want_rc 0 "matrix summary: a stack layer's skipped build passes" \
    env VALIDATE_RESULT=success BUILD_RESULT=skipped WEB_ONLY=false STACK_LAYER=true bash "$TMP/rbsum.sh"
  want_rc 1 "control: an UNEXPLAINED skipped build stays red" \
    env VALIDATE_RESULT=success BUILD_RESULT=skipped WEB_ONLY=false STACK_LAYER=false bash "$TMP/rbsum.sh"
  want_rc 1 "control: a FAILED build on a stack layer stays red" \
    env VALIDATE_RESULT=success BUILD_RESULT=failure WEB_ONLY=false STACK_LAYER=true bash "$TMP/rbsum.sh"
  want_rc 1 "control: being a stack layer does not excuse a failed validate" \
    env VALIDATE_RESULT=failure BUILD_RESULT=skipped WEB_ONLY=false STACK_LAYER=true bash "$TMP/rbsum.sh"
  # The inert-diff skip. Before this acceptance branch existed, ci-cost-controls
  # added the builds_binaries SKIP without teaching the summary that it was
  # legitimate -- so `dry-run summary` went red on exactly the diffs the
  # feature was built for. The pair below is the fix and its control.
  want_rc 0 "matrix summary: an inert diff's skipped build passes" \
    env VALIDATE_RESULT=success BUILD_RESULT=skipped WEB_ONLY=false STACK_LAYER=false BUILDS_BINARIES=false bash "$TMP/rbsum.sh"
  want_rc 1 "control: builds_binaries=true with a skipped build stays red" \
    env VALIDATE_RESULT=success BUILD_RESULT=skipped WEB_ONLY=false STACK_LAYER=false BUILDS_BINARIES=true bash "$TMP/rbsum.sh"

  # The classifier that decides builds_binaries, driven through its stdin mode.
  # GITHUB_OUTPUT must be PINNED: inside Actions it points at the step-output
  # file, so emit's lines vanish from the pipe and every row below reads rc=1
  # from grep. These passed locally (variable unset -> /dev/stdout) and failed
  # on the runner for exactly that reason.
  classify() { printf '%s\n' "$@" | env GITHUB_OUTPUT=/dev/stdout GITHUB_EVENT_NAME=pull_request bash .github/scripts/classify-diff.sh - 2>/dev/null | grep "^builds_binaries="; }
  # The wildcard (push/schedule/workflow_call) branch must emit ALL FOUR
  # outputs: under `set -u` a three-argument emit dies on unbound $4, and no
  # row exercised that branch at all.
  wildcard_classify() { env GITHUB_OUTPUT=/dev/stdout GITHUB_EVENT_NAME=schedule bash .github/scripts/classify-diff.sh 2>/dev/null | grep "^builds_binaries="; }
  want_rc_msg 0 "builds_binaries=false" "classify: a docs-only diff cannot change a binary" \
    classify docs/AUTOMERGER.md README.md
  want_rc_msg 0 "builds_binaries=true" "control: touching release-build.yml builds, .github or not" \
    classify docs/AUTOMERGER.md .github/workflows/release-build.yml
  want_rc_msg 0 "builds_binaries=true" "control: one crates/ file makes the whole diff build" \
    classify docs/AUTOMERGER.md crates/spark-model/src/lib.rs
  want_rc_msg 0 "builds_binaries=true" \
    "classify: a schedule event never fast-paths and emits all four outputs" \
    wildcard_classify
else
  bad "could not extract the dry-run summary shell from release-build.yml"
fi

echo "== one PR, one commit, one signer =="
extract 'pr-benchmark-gate/One PR, one commit, one signer' > "$TMP/oc.sh"
if [ -s "$TMP/oc.sh" ]; then
  R="$TMP/ocrepo"; rm -rf "$R"; mkdir -p "$R/.benchmarks/b"
  ( cd "$R" && git init -q . && git config user.email t@t && git config user.name t \
    && echo s > s && git add -A && git commit -qm base ) >/dev/null 2>&1
  BASE=$( cd "$R" && git rev-parse HEAD )
  mk() { printf '{"git_sha":"%s","recorded_at":1788300000}' "$2" > "$R/.benchmarks/b/$1.json"; }
  sg() { printf '{"v":1,"key":"%s","sig":"x"}' "$2" > "$R/.benchmarks/b/$1.json.sig"; }
  # ── What this step still does in bash ────────────────────────────────────
  # The commit/signer VERDICT moved into `gate::agreement` (Rust), because only
  # the registry knows a benchmark's Sensitivity and the rule is now
  # class-conditional. This selftest step is deliberately pure-Python with NO
  # Rust toolchain, so it cannot and must not run that half; the verdict has ten
  # tests with red/green controls under `cargo test -p atlas-plugin agreement`,
  # including the two that matter — a Speed set spanning signers is refused, a
  # Correctness set spanning signers is allowed.
  #
  # What remains here is the part that is still shell: collecting the records a
  # PR adds, and refusing an added record with no signature.
  mk r1 aaaaaaaaaa; sg r1 k1; ( cd "$R" && git add -A && git commit -qm r1 ) >/dev/null 2>&1
  # The backdating bypass: an ADDED record with no signature at all. `recorded_at`
  # lives INSIDE the file being authenticated, so backdating it below the signing
  # cutover would skip the signature entirely — git, not the file's own contents,
  # decides what a PR introduced.
  mk r2 aaaaaaaaaa
  ( cd "$R" && git add -A && git commit -qm r2 ) >/dev/null 2>&1
  want_rc_msg 1 "Unsigned record added" "control: an added record with no signature is refused" \
    sh -c "cd '$R' && BASE=$BASE bash '$TMP/oc.sh'"
  # WIRING: the step must still hand the verdict to the Rust gate. Without this
  # the class rule could be deleted from ci.yml and every test above would stay
  # green — the shell would simply stop asking.
  grep -q "record_agreement" "$TMP/oc.sh" \
    && ok "the step delegates the verdict to the Rust gate" \
    || bad "the step no longer invokes record_agreement — the commit/signer verdict is gone"
else
  bad "could not extract the one-commit-one-signer shell from ci.yml"
fi

echo "== certification-state.sh: the state machine =="
# Eleven states, chosen from the PR's merged flag, its mergeable_state, its
# queue entry, and three check-run conclusions. Driven here by a stubbed `gh`
# so every branch is reachable without a live PR.
mkstate() {  # mkstate <merged> <mergeable_state> <stamp> <seal> <records>
  mkdir -p "$TMP/bin"
  cat > "$TMP/bin/gh" <<STUB
#!/bin/bash
args="\$*"
case "\$args" in
  *"/pulls/"*"/merge"*) exit 1 ;;
  *check-runs*)
      # emit one line per matching check run, as --jq would
      # Print unconditionally. A guarded echo returns non-zero on an
      # empty value, which made the whole stub exit 1 and every "no such check
      # run" case look like an API failure rather than an absent check.
      case "\$args" in
        *Stamp*)  printf '%s\\n' "$3" ;;
        *Seal*)   printf '%s\\n' "$4" ;;
        *Certifications*) printf '%s\\n' "$5" ;;
      esac
      exit 0 ;;
  *"/pulls/"*) printf '{"merged":%s,"head":{"sha":"abc1234567"},"mergeable_state":"%s"}\n' "$1" "$2" ;;
  *) echo "" ;;
esac
STUB
  chmod +x "$TMP/bin/gh"
}
state_is() {  # state_is <expected> <label> <merged> <mergeable> <stamp> <seal> <records>
  local want=$1 label=$2; shift 2
  mkstate "$@"
  local got
  got=$(PATH="$TMP/bin:$PATH" REPO=o/r PREV_STATE="" bash .github/scripts/certification-state.sh 7 2>/dev/null | cut -f1)
  if [ "$got" = "$want" ]; then ok "$label"; else bad "$label (got '$got', want '$want')"; fi
}

state_is pr-certification-stage-1 "unstamped -> stage 1"                 false clean ""        ""        ""
state_is pr-certification-stage-2-both "stamped, nothing else -> stage 2 (both)" false clean success ""  ""
state_is pr-certification-stage-2-needs-records "stamped + seal -> needs records" false clean success success ""
state_is pr-certification-stage-2-needs-seal "stamped + records -> needs seal"   false clean success ""  success
state_is pr-certification-stage-3 "stamped + seal + records -> stage 3"  false clean success success success
state_is pr-certification-merged "a merged PR reads merged"              true  clean ""        ""        ""
state_is pr-certification-blocked "a conflicting PR reads blocked"       false dirty success success success

# CONTROL: precedence. A merged PR is merged whatever its checks say, and a
# conflicting one is blocked however green it is -- if either ever lost to the
# stage ladder, the bot would show a stale stage on a PR nobody can merge.
state_is pr-certification-merged "control: merged outranks a full stage-3 board" true clean success success success
state_is pr-certification-blocked "control: blocked outranks stage 3"    false dirty success success success
# CONTROL: a check that exists but did NOT succeed must not count as done.
state_is pr-certification-stage-2-both "control: a failed Seal is not a seal" false clean success failure ""
state_is pr-certification-stage-2-needs-seal "control: a failed Seal with records still needs a seal" false clean success failure success

# AN UNREADABLE API IS NOT A BOARD OF "NO"s. `pr_json=$(gh api ... || echo '{}')`
# and `conclusion_of`'s `2>/dev/null` made an outage look like a PR with nothing
# stamped, nothing sealed and no records. With a stage-2/3/queued marker already
# on the comment -- the bot's normal case -- the demotion branch then rendered
# "A new commit landed. The seal is void and the records no longer cover this
# tree — a perf path moved" over a matching diagram, and wrote that state into
# the marker, so the next run could not tell it had ever demoted. A fabricated
# event, posted as the PR's one status comment.
st_broken() {  # st_broken <which call fails: pulls|check-runs>
  mkdir -p "$TMP/bin"
  cat > "$TMP/bin/gh" <<STUB
#!/bin/bash
args="\$*"
case "\$args" in
  *check-runs*) [ "$1" = check-runs ] && { echo "gh: HTTP 502" >&2; exit 1; }; printf '\\n'; exit 0 ;;
  *"/pulls/"*)  [ "$1" = pulls ] && { echo "gh: HTTP 502" >&2; exit 1; }
                printf '{"merged":false,"head":{"sha":"abc1234567"},"mergeable_state":"clean"}\n'; exit 0 ;;
  *) echo "" ;;
esac
STUB
  chmod +x "$TMP/bin/gh"
}
st_run() {  # st_run -> "<exit>|<stdout>"
  local out rc
  out=$(PATH="$TMP/bin:$PATH" REPO=o/r PREV_STATE=pr-certification-stage-3 \
        bash .github/scripts/certification-state.sh 7 2>/dev/null); rc=$?
  printf '%s|%s' "$rc" "$out"
}
for which in pulls check-runs; do
  st_broken "$which"
  got=$(st_run)
  case "$got" in
    0\|*) bad "an unreadable $which API classified anyway, as '${got#*|}'" ;;
    *demoted*) bad "an unreadable $which API invented a demotion: ${got#*|}" ;;
    *) ok "an unreadable $which API refuses to classify, instead of inventing a demotion" ;;
  esac
done

# CONTROL: the pre-fix fallbacks, restored in a COPY, on the same broken API.
# They must produce the fabricated demotion -- that is what proves the two
# checks above are not passing by construction.
mkdir -p "$TMP/st"
cp .github/scripts/certification-state.sh "$TMP/st/state.sh"
python3 - "$TMP/st/state.sh" <<'STSAB'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text()
# Restore both pre-fix fallbacks: swallow the error and carry on with empties.
t = t.replace('''pr_json=$(gh api "repos/$REPO/pulls/$PR" 2>&1) || {''',
              '''pr_json=$(gh api "repos/$REPO/pulls/$PR" 2>/dev/null || echo '{"merged":false,"head":{"sha":"abc1234567"},"mergeable_state":"clean"}')
false && {''')
t = t.replace('''          --jq ".check_runs[] | select(.name == \\"$1\\") | .conclusion" 2>&1) || {''',
              '''          --jq ".check_runs[] | select(.name == \\"$1\\") | .conclusion" 2>/dev/null) || out=""
  false && {''')
t = t.replace('stamp=$(conclusion_of "Stamp") || exit 3', 'stamp=$(conclusion_of "Stamp")')
t = t.replace('seal=$(conclusion_of "Seal") || exit 3', 'seal=$(conclusion_of "Seal")')
t = t.replace('records=$(conclusion_of "PR Benchmark Certifications") || exit 3',
              'records=$(conclusion_of "PR Benchmark Certifications")')
p.write_text(t)
STSAB
st_broken check-runs
sab=$(PATH="$TMP/bin:$PATH" REPO=o/r PREV_STATE=pr-certification-stage-3 \
      bash "$TMP/st/state.sh" 7 2>/dev/null | cut -f1)
[ "$sab" = "pr-certification-demoted-push-both" ] \
  && ok "control: swallowing the API error invents 'the seal is void — a perf path moved'" \
  || bad "control: the sabotage did not reproduce the fabrication (got '$sab')"

echo "== command handlers: who may do what =="
# The handlers live in certification-commands.yml. What matters is not only that
# a refusal prints a message, but that a REFUSED command creates NO check run --
# a refusal that still mints the mark is cosmetic. The stub records every call so
# both halves can be asserted.
xcmd() { python3 -c '
import sys, yaml
d = yaml.safe_load(open(".github/workflows/certification-commands.yml"))
for j in d["jobs"].values():
    for st in j.get("steps", []) or []:
        if (st.get("name") or "") == sys.argv[1]:
            print(st["run"]); sys.exit(0)
' "$1"; }

ghspy() {  # records every gh invocation to $TMP/calls
  mkdir -p "$TMP/bin"; : > "$TMP/calls"
  cat > "$TMP/bin/gh" <<'STUB'
#!/bin/bash
printf '%s\n' "$*" >> "$CALLS"
for a in "$@"; do case "$a" in body=*|output*) printf '%s\n' "$a" >> "$CALLS.body";; esac; done
exit 0
STUB
  chmod +x "$TMP/bin/gh"
}
# made_check <name> -> did the handler POST a check run of that name?
made_check() { grep -q -- "-f name=$1" "$TMP/calls" 2>/dev/null; }
refused()    { grep -qi 'issues/.*/comments' "$TMP/calls" 2>/dev/null && ! made_check "$1"; }

xcmd '/expedite' > "$TMP/exp.sh"
if [ -s "$TMP/exp.sh" ]; then
  runexp() { ghspy; : > "$TMP/calls.body"
    ( PATH="$TMP/bin:$PATH" CALLS="$TMP/calls" REPO=o/r PR=1 ACTOR=u \
      PERM="$1" ARGS="$2" HEAD_SHA=abc1234567 SHORT=abc1234567 \
      bash "$TMP/exp.sh" >/dev/null 2>&1 ); }

  runexp admin "the GPU box is down for maintenance"
  made_check Expedite && ok "expedite: admin with a reason mints the waiver" \
    || bad "expedite: admin with a reason did NOT mint the waiver"
  grep -q 'maintenance' "$TMP/calls.body" 2>/dev/null && ok "expedite: the reason is recorded" \
    || bad "expedite: the reason was not recorded anywhere"

  # CONTROL: write access is NOT enough. /expedite discards the requirement to
  # prove anything, so it must be admin-only -- and the refusal must not mint.
  runexp write "trust me"
  refused Expedite && ok "control: write access cannot expedite" \
    || bad "control: write access expedited (or the refusal still minted the waiver)"
  runexp read "trust me"
  refused Expedite && ok "control: read access cannot expedite" \
    || bad "control: read access expedited"
  runexp none "trust me"
  refused Expedite && ok "control: a stranger cannot expedite" \
    || bad "control: a stranger expedited"

  # CONTROL: a reason is required. An unexplained bypass six months on is
  # indistinguishable from an accident.
  runexp admin ""
  refused Expedite && ok "control: admin without a reason is refused" \
    || bad "control: an unexplained expedite was accepted"
  runexp admin "   "
  refused Expedite && ok "control: whitespace is not a reason" \
    || bad "control: whitespace passed as a reason"
else
  bad "could not extract the /expedite handler"
fi

xcmd '/stamp and /seal' > "$TMP/ss.sh"
if [ -s "$TMP/ss.sh" ]; then
  runss() { ghspy
    ( PATH="$TMP/bin:$PATH" CALLS="$TMP/calls" REPO=o/r PR=1 ACTOR="$1" AUTHOR="$2" \
      VERB="$3" PERM="$4" HEAD_SHA=abc1234567 SHORT=abc1234567 \
      bash "$TMP/ss.sh" >/dev/null 2>&1 ); }

  runss alice alice /stamp none
  made_check Stamp && ok "stamp: the PR's author may stamp their own PR" \
    || bad "stamp: the author could not stamp their own PR"
  runss bob alice /stamp write
  made_check Stamp && ok "stamp: write access may stamp" || bad "stamp: write access could not stamp"
  # CONTROL: a stranger with no write access is neither.
  runss bob alice /stamp none
  refused Stamp && ok "control: a non-author without write cannot stamp" \
    || bad "control: a stranger stamped"
  # CONTROL: /seal is a different claim -- authorship confers nothing.
  runss alice alice /seal none
  refused Seal && ok "control: the author cannot seal without write access" \
    || bad "control: authorship was accepted as a seal"
else
  bad "could not extract the /stamp and /seal handler"
fi

echo "== seal status: the merge-queue path =="
# The seal job runs in the queue too, where there is no pull_request payload --
# the PR number has to be recovered from a GitHub-generated branch name. Getting
# this wrong either deadlocks the queue (nothing can land) or lets an unsealed
# entry through. It had only ever been exercised by hand.
xseal() { python3 -c '
import yaml
d = yaml.safe_load(open(".github/workflows/ci.yml"))
print(d["jobs"]["seal"]["steps"][0]["run"])
'; }
xseal > "$TMP/seal.sh"
if [ -s "$TMP/seal.sh" ]; then
  sealstub() {  # sealstub <check-runs-count> <pull-head-sha>
    mkdir -p "$TMP/bin"
    printf '#!/bin/bash\ncase "$*" in\n  *check-runs*) printf "%%s\\n" "%s"; exit 0 ;;\n  *pulls/*) printf "%%s\\n" "%s"; exit 0 ;;\n  *) exit 0 ;;\nesac\n' "$1" "$2" > "$TMP/bin/gh"
    chmod +x "$TMP/bin/gh"
  }
  runseal() {  # runseal <event> <pr_sha> <pr_num> <mq_ref>
    ( PATH="$TMP/bin:$PATH" REPO=o/r EVENT="$1" PR_SHA="$2" PR_NUM="$3" MQ_HEAD_REF="$4" \
      bash "$TMP/seal.sh" >"$TMP/sealout" 2>&1 ); }

  sealstub 1 abc1234567
  runseal pull_request abc1234567 7 ""; rc=$?
  [ "$rc" = 0 ] && ok "seal: a sealed PR passes" || bad "seal: a sealed PR did not pass (rc=$rc)"

  sealstub 0 abc1234567
  runseal pull_request abc1234567 7 ""; rc=$?
  [ "$rc" = 1 ] && ok "control: an unsealed PR fails" || bad "control: an unsealed PR passed (rc=$rc)"

  # A push to main has no PR to seal; holding it would block main forever.
  sealstub 0 ""
  runseal push "" "" ""; rc=$?
  [ "$rc" = 0 ] && ok "a push carries no PR and is not held" || bad "a push was held (rc=$rc)"

  # THE QUEUE PATH: recover the PR from gh-readonly-queue/<base>/pr-<N>-<sha>.
  sealstub 1 abc1234567
  runseal merge_group "" "" "refs/heads/gh-readonly-queue/main/pr-840-abc1234567"; rc=$?
  [ "$rc" = 0 ] && ok "queue: a sealed entry passes" || bad "queue: a sealed entry failed (rc=$rc)"
  # The success path prints only the sha, so the recovered number is asserted on
  # the FAILURE path, which names "PR #<n>". That proves the number was derived
  # from the branch AND that the right PR was consulted -- a wrong number would
  # look up someone else's seal.
  sealstub 0 abc1234567
  runseal merge_group "" "" "refs/heads/gh-readonly-queue/main/pr-840-abc1234567"
  grep -q '#840' "$TMP/sealout" 2>/dev/null && ok "queue: the PR number is recovered from the branch" \
    || bad "queue: the PR number was not recovered from the branch name"

  sealstub 0 abc1234567
  runseal merge_group "" "" "refs/heads/gh-readonly-queue/main/pr-840-abc1234567"; rc=$?
  [ "$rc" = 1 ] && ok "control: an unsealed queue entry is ejected" || bad "control: an unsealed queue entry passed (rc=$rc)"

  # CONTROL: an unparseable queue branch must REFUSE, not guess. Guessing here
  # would either seal the wrong PR or wave through an unknown one.
  sealstub 1 abc1234567
  runseal merge_group "" "" "refs/heads/gh-readonly-queue/main/garbage"; rc=$?
  [ "$rc" = 1 ] && ok "control: an unparseable queue branch is refused" || bad "control: an unparseable branch was accepted (rc=$rc)"

  # CONTROL: fail CLOSED. This inverts the stamp job beside it on purpose -- a
  # stamp only spends runners, but a seal is a person vouching, and an
  # unreadable API must never be read as a vouch nobody gave.
  mkdir -p "$TMP/bin"; printf '#!/bin/bash\nexit 1\n' > "$TMP/bin/gh"; chmod +x "$TMP/bin/gh"
  runseal pull_request abc1234567 7 ""; rc=$?
  # rc=1 alone is NOT enough here. With the fail-closed branch deleted the script
  # still exits 1 -- by accident, via `[: REFUSE: integer expression expected`
  # falling through to the generic "Not sealed" message. That is safe today and
  # fragile tomorrow, and it tells the operator the wrong thing: "no seal" when
  # the truth is "we could not look". So pin the DELIBERATE path by its message.
  if [ "$rc" = 1 ] && grep -q 'Could not read the seal' "$TMP/sealout"; then
    ok "control: an unreadable API fails CLOSED, deliberately"
  else
    bad "control: unreadable API -> rc=$rc, and not via the fail-closed branch"
    sed 's/^/       /' "$TMP/sealout" | head -3
  fi
else
  bad "could not extract the seal job's shell from ci.yml"
fi

echo "== command authority is bounded =="
want_rc 0 "only state-changing commands can change state" \
  python3 .github/scripts/assert-command-authority.py
# CONTROL x3: each way a text-only command could quietly gain authority.
mkdir -p "$TMP/auth/.github/workflows"; cp .github/scripts/assert-command-authority.py "$TMP/auth/"
mkwf() {  # mkwf <extra-run-line-for-/review> <permissions-yaml>
  cat > "$TMP/auth/.github/workflows/certification-commands.yml" <<WF
name: c
on: { issue_comment: { types: [created] } }
permissions:
$2
jobs:
  cmd:
    runs-on: ubuntu-latest
    steps:
      - name: /review
        run: |
          echo reviewing
          $1
      - name: /stamp and /seal
        run: gh api -X POST "repos/o/r/check-runs" -f name=Stamp
WF
}
mkwf 'echo ok' '  contents: read'
want_rc 0 "a text-only /review passes" sh -c "cd '$TMP/auth' && python3 assert-command-authority.py"
mkwf 'gh api -X POST "repos/o/r/check-runs" -f name=Seal' '  contents: read'
want_rc 1 "control: /review minting a check run is refused" \
  sh -c "cd '$TMP/auth' && python3 assert-command-authority.py"
mkwf 'gh api -X POST "repos/o/r/actions/runs/1/rerun"' '  contents: read'
want_rc 1 "control: /review re-running CI is refused" \
  sh -c "cd '$TMP/auth' && python3 assert-command-authority.py"
mkwf 'echo ok' '  contents: write
  checks: write'
want_rc 1 "control: widened workflow permissions are refused" \
  sh -c "cd '$TMP/auth' && python3 assert-command-authority.py"

echo "== /review refuses safely =="
# Every refusal here exits 0 on purpose: a /review that cannot reach a model
# must not fail anyone's CI. That makes the exit code useless as an assertion,
# so each check pins the MESSAGE -- and, for the two missing-config cases, that
# no request was attempted at all.
revstub() {  # revstub <http-code> <response-body>
  mkdir -p "$TMP/bin"; : > "$TMP/rcalls"; : > "$TMP/curled"
  # pr-review.sh posts with `-F body=@-`, i.e. the message arrives on STDIN, not
  # in argv. A stub that only records arguments captures the call and loses the
  # text -- which read as "the script never explained itself" when it had.
  cat > "$TMP/bin/gh" <<'STUB'
#!/bin/bash
printf 'ARGV: %s\n' "$*" >> "$RCALLS"
case "$*" in *body=@-*) cat >> "$RCALLS" ;; esac
exit 0
STUB
  cat > "$TMP/bin/curl" <<STUB
#!/bin/bash
echo called >> "\$CURLED"
for a in "\$@"; do case "\$a" in /tmp/or.json) : ;; esac; done
printf '%s' '$2' > /tmp/or.json
printf '%s' '$1'
STUB
  chmod +x "$TMP/bin/gh" "$TMP/bin/curl"
}
runrev() {
  ( PATH="$TMP/bin:$PATH" RCALLS="$TMP/rcalls" CURLED="$TMP/curled" \
    REPO=o/r PR=1 ACTOR=u ARGS="" SHORT=abc1234567 \
    OPENROUTER_KEY="$1" OPENROUTER_DEFAULT_FREE_MODEL="$2" \
    bash .github/scripts/pr-review.sh >/dev/null 2>&1 ); echo $?; }

revstub 200 '{"choices":[{"message":{"content":"looks fine"}}]}'
rc=$(runrev "" "nvidia/x:free")
[ "$rc" = 0 ] && grep -qi 'key' "$TMP/rcalls" && ok "no API key: says so, and does not fail CI" \
  || bad "no API key: rc=$rc, or it never explained itself"
# CONTROL: with no key it must not attempt a request.
[ ! -s "$TMP/curled" ] && ok "control: no key -> no request attempted" \
  || bad "control: it called the endpoint without a key"

revstub 200 '{"choices":[{"message":{"content":"looks fine"}}]}'
rc=$(runrev "k" "")
[ "$rc" = 0 ] && grep -q 'OPENROUTER_DEFAULT_FREE_MODEL' "$TMP/rcalls" \
  && ok "no model id: names the variable that is empty" \
  || bad "no model id: rc=$rc, or it did not name the variable"
[ ! -s "$TMP/curled" ] && ok "control: no model -> no request attempted" \
  || bad "control: it called the endpoint with no model"

revstub 500 '{}'
rc=$(runrev "k" "nvidia/x:free")
[ "$rc" = 0 ] && grep -qi 'could not reach the model' "$TMP/rcalls" \
  && ok "a 500 is reported, not swallowed" || bad "a 500 was not reported (rc=$rc)"
grep -q '500' "$TMP/rcalls" && ok "the HTTP code is in the message" || bad "the HTTP code was not reported"

revstub 200 '{"choices":[{"message":{"content":"THE REVIEW BODY"}}]}'
rc=$(runrev "k" "nvidia/x:free")
grep -q 'THE REVIEW BODY' "$TMP/rcalls" && ok "a 200 posts the model's answer" \
  || bad "a 200 did not post the answer"

# The PR's own prose is attacker-controlled. It must be fenced so a title or
# body cannot issue instructions to the model.
grep -q 'UNTRUSTED' .github/scripts/pr-review.sh \
  && ok "untrusted PR prose is fenced in the prompt" \
  || bad "PR prose is not fenced -- a title could instruct the model"

echo "== the bot's one comment, and the certificate =="
# The bot keeps ONE comment and edits it in place, except on merge, where it
# posts a second one -- because GitHub does not notify on an @mention added by
# EDITING a comment, so tagging the authors in the edited comment would alert
# nobody. Both halves are asserted here, plus the idempotency that stops a
# contributor being tagged four times.
xbot() { python3 -c '
import sys, yaml
d = yaml.safe_load(open(".github/workflows/certification-bot.yml"))
for j in d["jobs"].values():
    for st in j.get("steps", []) or []:
        if (st.get("name") or "") == sys.argv[1]:
            print(st["run"]); sys.exit(0)
' "$1"; }
xbot 'Post or edit the one comment' > "$TMP/bot.sh"
if [ -s "$TMP/bot.sh" ]; then
  botstub() {  # botstub <existing-certificate-count>
    mkdir -p "$TMP/bin"; : > "$TMP/bcalls"
    cat > "$TMP/bin/gh" <<STUB
#!/bin/bash
printf 'ARGV: %s\\n' "\$*" >> "\$BCALLS"
case "\$*" in
  *comments*--jq*) echo "$1" ;;                       # certificate-marker count
  *"/pulls/"*commits*) echo "alice" ;;
  *"/pulls/"*) echo "alice" ;;
  *) : ;;
esac
exit 0
STUB
    chmod +x "$TMP/bin/gh"
    printf '#!/bin/bash\nexit 0\n' > "$TMP/bin/sudo"; chmod +x "$TMP/bin/sudo"
    # rsvg-convert must actually PRODUCE its output. A no-op stub left no PNG,
    # so the sha256sum that names the file failed and `set -euo pipefail` aborted
    # the step before it ever posted -- which read as "the bot does not post a
    # certificate on merge" when the bot was right and the stub was hollow.
    cat > "$TMP/bin/rsvg-convert" <<'STUB'
#!/bin/bash
out=""; while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift 2;; *) shift;; esac; done
[ -n "$out" ] && printf 'PNG-STUB' > "$out"
exit 0
STUB
    chmod +x "$TMP/bin/rsvg-convert"
  }
  runbot() {  # runbot <state> <comment_id> <cert-count>
    botstub "$3"
    ( PATH="$TMP/bin:$PATH" BCALLS="$TMP/bcalls" REPO=o/r PR=1 DEFAULT_BRANCH=main \
      STATE="$1" HEADLINE=h COMMENT_ID="$2" HEAD_SHA=abc1234567 \
      bash "$TMP/bot.sh" >/dev/null 2>&1 ); }
  patched() { grep -q 'PATCH' "$TMP/bcalls"; }
  posted()  { grep -q 'POST.*issues/1/comments' "$TMP/bcalls"; }
  # The idempotency QUERY also contains the literal marker, inside its --jq
  # filter. Grepping for the marker alone matched the lookup and reported a
  # certificate that was never posted -- the assertion could not tell "asked
  # whether one exists" from "posted one". Require the POST too.
  certed()  { grep -qE 'POST.*issues/1/comments.*atlas-certificate' "$TMP/bcalls"; }

  # CONTROL: the render tools are installed with `|| true`, so ask what happens
  # when that install fails. Before the guard, `rsvg-convert` was then missing,
  # the step died under `set -e` BEFORE the comment POST, and a merged PR got no
  # certificate and no comment at all -- strictly worse than the generic image
  # the fallback exists to serve.
  #
  # The host running this suite HAS both tools, so their absence is manufactured
  # rather than assumed: a PATH of the stubs plus a symlink farm of the commands
  # the step actually uses, resolved at runtime so it adapts to wherever they
  # live. Asserting "the host happens not to have librsvg" is the wave-2
  # mistake -- a control that only holds on one machine.
  mkdir -p "$TMP/farm"
  for c in bash sh env python3 jq sed awk grep tr cut date base64 sha256sum \
           cat head tail printf dirname basename mktemp sort uniq wc xargs; do
    src=$(command -v "$c" 2>/dev/null) && ln -sf "$src" "$TMP/farm/$c"
  done
  if [ -e "$TMP/farm/rsvg-convert" ]; then
    bad "control setup: the farm leaked rsvg-convert; absence was not manufactured"
  else
    botstub 0
    rm -f "$TMP/bin/rsvg-convert"
    ( PATH="$TMP/bin:$TMP/farm" BCALLS="$TMP/bcalls" REPO=o/r PR=1 DEFAULT_BRANCH=main \
      STATE=pr-certification-merged HEADLINE=h COMMENT_ID= HEAD_SHA=abc1234567 \
      bash "$TMP/bot.sh" >/dev/null 2>&1 )
    if certed; then
      ok "control: with no rsvg-convert the certificate is still posted"
    else
      bad "control: a missing render tool swallowed the certificate entirely"
    fi
    # It must NOT fall back to the committed template. That sample carries the
    # placeholder certifier names it was designed with (m-ferraro / a-hoffmann)
    # over a fixed author line, so linking it tells the world that fictional
    # people certified this PR. Observed on #901 (opened by @TheTom, sample says
    # tbraun96), and on every certificate before it: bot-cards held zero pr-*.png.
    if grep -q 'certificate-merged.png' "$TMP/bcalls"; then
      bad "control: the fallback linked the template — that publishes placeholder certifiers"
    else
      ok "control: no render, no image — the template's placeholder names are never published"
    fi
    if grep -q 'could not be rendered' "$TMP/bcalls"; then
      ok "control: and it says why the picture is missing"
    else
      bad "control: the image vanished with no explanation"
    fi
  fi

  # The certificate IMAGE only exists if the PNG reaches bot-cards, and that
  # PUT needs contents:write. The App did not have it (certification-preflight's
  # permission probe fails), which is why bot-cards held zero pr-*.png and every
  # certificate went imageless. The job's own grant is the guarantee, so pin it.
  if python3 - <<'PY'
import sys, yaml
d = yaml.safe_load(open(".github/workflows/certification-bot.yml"))
perms = (d["jobs"]["state"].get("permissions") or {})
sys.exit(0 if perms.get("contents") == "write" else 1)
PY
  then
    ok "the bot job grants itself contents:write — the certificate image can be uploaded"
  else
    bad "the bot job lacks contents:write — the certificate PNG can never reach bot-cards"
  fi
  # ...and the upload must actually FALL BACK to that token, not rely on the App.
  if grep -q 'put_cert "${WORKFLOW_TOKEN:-}"' .github/workflows/certification-bot.yml; then
    ok "and the upload retries under the workflow token when the App lacks the grant"
  else
    bad "the upload never falls back to the token that is guaranteed to have contents:write"
  fi

  runbot pr-certification-stage-1 "" 0
  posted && ! patched && ok "no prior comment -> posts one" || bad "no prior comment -> did not post exactly one"
  # CONTROL: with a prior comment it must EDIT, never post a second. A thread of
  # stale state comments is precisely what the marker exists to prevent.
  runbot pr-certification-stage-1 12345 0
  patched && ok "control: an existing comment is edited, not duplicated" \
    || bad "control: it posted a second state comment instead of editing"

  # The marker is both the lookup key and the memory of the previous state, and
  # BOTH halves live in the state it carries. A substring grep of the step's
  # source could not see that: deleting the `:$STATE` leaves the string
  # `atlas-certification-state` in the file, so this check stayed green while
  # the lookup's `contains("<!-- atlas-certification-state:")` matched nothing,
  # `prev` was empty on every run, and the bot posted a fresh comment per event
  # instead of editing one -- the thread of stale states the marker exists to
  # prevent. Assert the marker the bot actually EMITS, and that the lookup's own
  # literal is a prefix of it. (The nearby "edited, not duplicated" control
  # cannot catch this either: it feeds COMMENT_ID in, bypassing the lookup.)
  LOOKUP=$(python3 - <<'PY'
import re, pathlib
t = pathlib.Path(".github/workflows/certification-bot.yml").read_text()
m = re.search(r'contains\("(<!-- atlas-certification-state[^"]*)"\)', t)
print(m.group(1) if m else "")
PY
)
  runbot pr-certification-stage-3 "" 0
  if [ -z "$LOOKUP" ]; then
    bad "the bot has no marker lookup -- it cannot find its own comment at all"
  elif grep -qF "${LOOKUP}pr-certification-stage-3 -->" "$TMP/bcalls"; then
    ok "the state marker is written into the comment, and matches the bot's own lookup"
  else
    bad "the posted marker does not match the lookup '$LOOKUP' plus the state -- prev can never be read back"
  fi

  # THE CERTIFICATE: only on merge, and only once.
  runbot pr-certification-merged "" 0
  certed && ok "merged -> the certificate is posted" || bad "merged -> no certificate"
  runbot pr-certification-stage-3 "" 0
  ! certed && ok "control: a non-merged state posts no certificate" \
    || bad "control: a certificate was posted for a PR that has not merged"
  # CONTROL: the bot also fires on check_run completions AFTER a merge. Without
  # the marker check a contributor is tagged once per event.
  runbot pr-certification-merged "" 1
  ! certed && ok "control: a certificate already posted is not posted again" \
    || bad "control: it posted a second certificate"

  # CONTROL: the PR's title and merge sha are read with `gh api ... 2>/dev/null`,
  # so an unanswered API is indistinguishable from an answer of "". The renderer
  # now REFUSES those rather than drawing the template's specimen title and
  # commit in their place -- and it refuses by exiting non-zero, which under
  # `set -e` would kill this step before the certificate POST. A merged PR with
  # no certificate at all is strictly worse than one with no picture, which is
  # the same trap the rsvg-convert guard exists for. So: the step must ask
  # first, skip the render, and still post.
  botstub 0
  cat > "$TMP/bin/gh" <<'STUB'
#!/bin/bash
printf 'ARGV: %s\n' "$*" >> "$BCALLS"
case "$*" in
  *comments*--jq*)      echo 0 ;;        # no certificate posted yet
  *"/pulls/1/commits"*) echo "alice" ;;
  *"/pulls/1"*)         : ;;             # title, merge sha and opener: no answer
  *)                    : ;;
esac
exit 0
STUB
  chmod +x "$TMP/bin/gh"
  ( PATH="$TMP/bin:$PATH" BCALLS="$TMP/bcalls" REPO=o/r PR=1 DEFAULT_BRANCH=main \
    STATE=pr-certification-merged HEADLINE=h COMMENT_ID= HEAD_SHA=abc1234567 \
    bash "$TMP/bot.sh" >/dev/null 2>&1 )
  certed && ok "control: an unreadable PR title costs the picture, not the certificate" \
    || bad "control: an empty title killed the step before the certificate was posted"
  grep -q 'PUT.*contents/pr-1-' "$TMP/bcalls" \
    && bad "control: it rendered and uploaded a certificate whose title it never read" \
    || ok "control: and no image is rendered from data the API never returned"

  # CONTROL: when the image upload fails, the comment must NOT link an object
  # that was never written. #843 shipped a certificate whose <img> was a 404,
  # because the PUT is non-fatal by design and nothing checked the result.
  # The stub's contents lookup fails, so the fallback must be chosen.
  botstub 0
  cat > "$TMP/bin/gh" <<'STUB'
#!/bin/bash
printf 'ARGV: %s\n' "$*" >> "$BCALLS"
case "$*" in
  *contents*bot-cards*) exit 1 ;;                 # the image is NOT there
  *comments*--jq*) echo "0" ;;
  *"/pulls/"*) echo "alice" ;;
  *) : ;;
esac
exit 0
STUB
  chmod +x "$TMP/bin/gh"
  ( PATH="$TMP/bin:$PATH" BCALLS="$TMP/bcalls" REPO=o/r PR=1 DEFAULT_BRANCH=main \
    STATE=pr-certification-merged HEADLINE=h COMMENT_ID= HEAD_SHA=abc1234567 \
    bash "$TMP/bot.sh" >/dev/null 2>&1 )
  if grep -qE 'POST.*issues/1/comments.*atlas-certificate' "$TMP/bcalls"; then
    if grep -q 'bot-cards/pr-1-' "$TMP/bcalls"; then
      bad "control: it linked an image that was never uploaded"
    else
      ok "control: a failed upload falls back, never links a 404"
    fi
  else
    bad "control: no certificate was posted at all when the upload failed"
  fi
else
  bad "could not extract the bot's comment step"
fi

# ---------------------------------------------------------------------------
# Jobs that report a verdict nobody consults (#810, and the ancestry guard)
# ---------------------------------------------------------------------------
# Two instances of one defect. `Site unit tests` ran on every PR and blocked
# neither merge nor deploy. And `Merge-ancestry guard self-test` -- the test
# proving the guard CAN fail -- was required, while the guard's actual verdict
# on your branch was not.
#
# The guard is hosted here, in a required context, rather than in either
# workflow's own lane: a job cannot be relied on to notice it has been unwired
# from itself.
#
# Every control pins the *message* as well as the exit code. Six distinct
# defects leave through the same exit 1, so asserting rc alone asserts nothing
# about which one fired -- the wave 4 and wave 7 lesson.
want_rc 0 "both gated jobs are wired as things stand" \
  python3 .github/scripts/assert-gates-are-wired.py
mkdir -p "$TMP/sg/scripts" "$TMP/sg/workflows"
cp .github/scripts/assert-gates-are-wired.py "$TMP/sg/scripts/"

# $1 = workflow filename, $2 = python edit applied to the parsed doc as `d`.
# NOTE `d[True]`, not `d["on"]`: YAML 1.1 parses a bare `on:` key as the boolean
# True. An edit reaching for the string raises KeyError, the sabotage silently
# does not land, and the control then passes against unmodified input -- which
# is a green light that measured nothing. Caught exactly that way once already.
sg_sabotage() {
  # EVERY workflow, not just the sabotaged one: the assertions read across
  # files (ci.yml calls release-build.yml, and the required-context list
  # spans nine of them). Staging a subset made the script die with
  # FileNotFoundError -- rc=1 for the wrong reason, which want_rc_msg
  # catches only because it also pins the message.
  for w in .github/workflows/*.yml; do cp "$w" "$TMP/sg/workflows/"; done
  python3 - "$TMP/sg/workflows/$1" "$2" <<'PY'
import pathlib, sys, yaml
p = pathlib.Path(sys.argv[1]); d = yaml.safe_load(p.read_text())
exec(sys.argv[2], {"d": d})
p.write_text(yaml.safe_dump(d, sort_keys=False))
PY
}

sg_sabotage site.yml 'd["jobs"]["deploy"]["needs"] = ["build"]'
want_rc_msg 1 "does not need" "control: dropping unit from deploy's needs is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

sg_sabotage site.yml 'd["jobs"]["unit"]["if"] = "github.event_name == \"push\""'
want_rc_msg 1 "grew an \`if:\`" "control: making the site suite conditional is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

sg_sabotage site.yml 'd[True]["pull_request"] = {"branches": ["main"], "paths": ["site/**"]}'
want_rc_msg 1 "grew a \`paths:\` filter" "control: a paths filter that would deadlock PRs is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

sg_sabotage site.yml 'd["jobs"].pop("unit")'
want_rc_msg 1 "no longer exists" "control: deleting the site suite outright is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

# The ancestry guard's own regression: restoring the `if:` that made its
# verdict advisory while its self-test stayed required.
sg_sabotage merge-ancestry.yml 'd["jobs"]["guard"]["if"] = "github.event_name == \"pull_request\""'
want_rc_msg 1 "must report on every run" "control: re-adding the ancestry guard's if: is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

sg_sabotage merge-ancestry.yml 'd[True].pop("merge_group")'
want_rc_msg 1 "every entry would deadlock" "control: dropping merge_group support is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

# A rename is the quietest way to lose a required context: the job still runs,
# still passes, and reports under a name branch protection is not waiting for.
sg_sabotage merge-ancestry.yml 'd["jobs"]["guard"]["name"] = "Ancestry"'
want_rc_msg 1 "leaves the required context uncreated" "control: renaming a required job is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

# ---------------------------------------------------------------------------
# A required check whose verdict is not about the thing it claims to test
# ---------------------------------------------------------------------------
# `cargo test --features metal (macOS aarch64)` inherited ci.yml's
# workflow-level ATLAS_SKIP_BUILD=1 (there so the ubuntu jobs type-check
# without nvcc). atlas-kernels' build.rs honours it first and emits a stub
# whose `metallib_modules()` is Vec::new(), so MetalGpuBackend loaded ZERO
# libraries and all 35 parity tests died with `Metal: unknown module`. The
# check was permanently red about a stub, and the merge queue was impassable
# for every non-web PR. Same family as a check that passes vacuously: the
# verdict is not about the capability named on the tin.
#
# Staged with the same sg_sabotage helper above -- it copies every workflow,
# which these assertions need.
#
# The metal step's env, addressed by name so a sabotage that does not land is
# an AssertionError rather than a control that silently measured nothing.
rc_metal_env='
steps = [s for s in d["jobs"]["test-macos-metal"]["steps"] if s.get("name","").startswith("cargo test -p spark-runtime")]
assert len(steps) == 1, f"sabotage would not land: {len(steps)} metal test steps"
env = steps[0]["env"]
'

want_rc 0 "every required context resolves to a live job that a failed dependency cannot skip" \
  python3 .github/scripts/assert-gates-are-wired.py

# The regression itself: put the stub env back on the metal test step.
sg_sabotage ci.yml "$rc_metal_env"'env["ATLAS_SKIP_BUILD"] = "1"'
want_rc_msg 1 "is about the stub, not the kernels" \
  "control: running the metal suite against a kernel-build stub is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

# Inheritance, not just the step: deleting the step override lets ci.yml's
# workflow-level ATLAS_SKIP_BUILD=1 reach the job again. A guard that only
# looked at the step's own env would pass here.
sg_sabotage ci.yml "$rc_metal_env"'env.pop("ATLAS_SKIP_BUILD")'
want_rc_msg 1 "is about the stub, not the kernels" \
  "control: dropping the override so the workflow-level stub env is inherited is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

# Without ATLAS_TARGET_HW, build.rs takes its macOS auto-skip and embeds
# nothing even with ATLAS_SKIP_BUILD=0 -- the same empty set by another route.
sg_sabotage ci.yml "$rc_metal_env"'env.pop("ATLAS_TARGET_HW")'
want_rc_msg 1 "build.rs takes the macOS auto-skip" \
  "control: dropping ATLAS_TARGET_HW from the metal suite is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

# The other half of the family: a required job that a FAILED dependency
# silently skips. Branch protection counts a skipped check as satisfied, so
# the gate reports safety it never measured. This is the exact pre-fix state
# of docs.yml `build` -- `needs: [changes]` with no `if:` at all.
sg_sabotage docs.yml 'd["jobs"]["build"].pop("if")'
want_rc_msg 1 "would pass vacuously" \
  "control: a required job a failed dependency can skip is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

# Renames and deletes across the whole required list, not just the two GATES.
sg_sabotage security.yml 'd["jobs"]["cargo-deny"]["name"] = "deny"'
want_rc_msg 1 "a renamed job leaves the required context uncreated" \
  "control: renaming cargo deny is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

sg_sabotage gdn-so-pin.yml 'd["jobs"].pop("verify-gdn-so-pins")'
want_rc_msg 1 "every PR would hang" \
  "control: deleting the GDN pin job is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

# The nested reusable-workflow context. `release matrix / dry-run summary` is
# TWO names -- the caller's and the callee's -- and renaming either one leaves
# protection waiting on a string nothing emits.
sg_sabotage release-build.yml 'd["jobs"]["dry-run-summary"]["name"] = "summary"'
want_rc_msg 1 "would never be created" \
  "control: renaming the nested dry-run summary job is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

sg_sabotage ci.yml 'd["jobs"]["release-matrix"]["uses"] = "./.github/workflows/release.yml"'
want_rc_msg 1 "no longer calls release-build.yml" \
  "control: repointing the release-matrix caller is caught" \
  python3 "$TMP/sg/scripts/assert-gates-are-wired.py"

# ---------------------------------------------------------------------------
# A grep-based gate that scanned nothing and said OK
# ---------------------------------------------------------------------------
# `No block_on under tui/ or recipe/` is a required context implemented as
# `if hits=$(grep -r ... dirs 2>/dev/null); then fail; fi`. On a MISSING
# directory grep exits 2, the redirect hides the reason, and the `if` reads any
# non-zero as "no hits" -- so the check printed OK and exited 0 against a tree
# with no tui/ at all. Verified in an empty directory before the fix. A rename
# of either tree would have retired the rule while the check stayed green.
#
# Run the step's REAL shell, extracted from the workflow, so the control cannot
# drift from what CI executes.
mkdir -p "$TMP/bo"
python3 - "$TMP/bo/step.sh" "$TMP/bo/scan_dirs" <<'PY'
import pathlib, sys, yaml
d = yaml.safe_load(pathlib.Path(".github/workflows/tui-threading.yml").read_text())
job = d["jobs"]["no-blocking-on-the-render-thread"]
steps = [s for s in job["steps"] if s.get("name", "").startswith("Check the render thread")]
assert len(steps) == 1, f"expected one render-thread step, found {len(steps)}"
pathlib.Path(sys.argv[1]).write_text(steps[0]["run"])
pathlib.Path(sys.argv[2]).write_text(steps[0]["env"]["SCAN_DIRS"])
PY
BO_DIRS=$(cat "$TMP/bo/scan_dirs")

want_rc 0 "the render-thread gate passes on the real tree" \
  env SCAN_DIRS="$BO_DIRS" bash "$TMP/bo/step.sh"

# A tree where one scanned directory does not exist. Pre-fix this printed
# "OK: no block_on/block_in_place under tui/ or recipe/" and exited 0.
mkdir -p "$TMP/bo/moved/crates/spark-server/src/recipe"
want_rc_msg 1 "does not exist, so this check scanned nothing" \
  "control: the render-thread gate refuses a tree it cannot scan" \
  env -C "$TMP/bo/moved" SCAN_DIRS="$BO_DIRS" bash "$TMP/bo/step.sh"

# ...and the existence assertion did not neuter the rule it guards: a real
# block_on in a present tree is still caught.
mkdir -p "$TMP/bo/dirty/crates/spark-server/src/tui" "$TMP/bo/dirty/crates/spark-server/src/recipe"
printf 'fn f() { handle.block_on(fut); }\n' > "$TMP/bo/dirty/crates/spark-server/src/tui/draw.rs"
want_rc_msg 1 "must never poll a future" \
  "control: the render-thread gate still catches a real block_on" \
  env -C "$TMP/bo/dirty" SCAN_DIRS="$BO_DIRS" bash "$TMP/bo/step.sh"

# The same defect twice more, in the two other gates that decide by scanning a
# tree. Both were verified passing on a tree they could not see.
#
# `Enforce <=500 LoC per source file`: `find crates ...` on a missing directory
# feeds the loop nothing, `violations` stays 0, and the step printed its tick
# and exited 0 -- the cap unenforced repo-wide. `set -euo pipefail` does not
# catch it: the find lives in a process substitution whose status is not the
# command's.
mkdir -p "$TMP/fs"
python3 - "$TMP/fs/step.sh" <<'PY'
import pathlib, sys, yaml
d = yaml.safe_load(pathlib.Path(".github/workflows/file-size-cap.yml").read_text())
steps = [s for s in d["jobs"]["check-file-sizes"]["steps"]
         if s.get("name", "").startswith("Check no .rs file")]
assert len(steps) == 1, f"expected one file-size step, found {len(steps)}"
pathlib.Path(sys.argv[1]).write_text(steps[0]["run"])
PY
want_rc 0 "the 500-LoC cap passes on the real tree" bash "$TMP/fs/step.sh"

mkdir -p "$TMP/fs/nocrates"
want_rc_msg 1 "the 500-line cap scanned nothing" \
  "control: the 500-LoC cap refuses a tree with no crates/ to scan" \
  env -C "$TMP/fs/nocrates" bash "$TMP/fs/step.sh"

# `kernel shadow structure`: the scan loop skipped any hardware tree that was
# not present, so an empty (or renamed) kernels/ printed
# "kernel shadow structure: OK" and exited 0. HW_SOURCE_EXT is the list of
# trees the check CLAIMS to cover; every one of them must be there.
want_rc 0 "the kernel-shadow check passes on the real kernels/ tree" \
  python3 scripts/check_kernel_shadows.py

mkdir -p "$TMP/ks/kernels"
want_rc_msg 1 "is missing hardware tree(s)" \
  "control: the kernel-shadow check refuses a kernels/ tree it cannot scan" \
  python3 scripts/check_kernel_shadows.py "$TMP/ks/kernels"

# ...and it still catches the violation it exists for: a shadow byte-identical
# to the common file it overrides is a dead override.
mkdir -p "$TMP/ks/live/gb10/common" "$TMP/ks/live/gb10/m1/q" \
         "$TMP/ks/live/metal" "$TMP/ks/live/strix" "$TMP/ks/live/strix-hip"
printf '__global__ void k() {}\n' > "$TMP/ks/live/gb10/common/k.cu"
cp "$TMP/ks/live/gb10/common/k.cu" "$TMP/ks/live/gb10/m1/q/k.cu"
want_rc_msg 1 "RULE1" "control: the kernel-shadow check still catches a dead override" \
  python3 scripts/check_kernel_shadows.py "$TMP/ks/live"

# ---------------------------------------------------------------------------
# A write whose failure is swallowed, and no probe behind it
# ---------------------------------------------------------------------------
# Shipped twice. #843: the certificate image never uploaded because
# contents:write was absent, the PUT is non-fatal, and the preflight probed
# contents:READ. #847: /stamp recorded its mark and did not re-run the held
# lane, because actions:write was absent, the re-run's stderr went to
# /dev/null, and the preflight probed actions:READ -- under a justification
# that read "/stamp re-runs the held CI run".
want_rc 0 "every suppressed write has a real probe behind it" \
  python3 .github/scripts/assert-preflight-covers-writes.py
mkdir -p "$TMP/pw/scripts" "$TMP/pw/workflows"
cp .github/scripts/assert-preflight-covers-writes.py "$TMP/pw/scripts/"

pw_sabotage() {  # $1 = permission whose probe is deleted from the preflight copy
  for w in certification-preflight.yml certification-commands.yml certification-bot.yml; do
    cp ".github/workflows/$w" "$TMP/pw/workflows/$w"
  done
  python3 - "$TMP/pw/workflows/certification-preflight.yml" "$1" <<'PY'
import pathlib, sys, re
p = pathlib.Path(sys.argv[1]); t = p.read_text()
n = t.count(f'probe_write "{sys.argv[2]}"')
assert n == 1, f"sabotage would not land: {n} probe_write for {sys.argv[2]}"
p.write_text(re.sub(r'probe_write\s+"%s"' % re.escape(sys.argv[2]), 'probe "%s"' % sys.argv[2], t))
PY
}

# Downgrading the probe to a read-style `probe` is the exact shape of both real
# defects -- the call is still made, so a grep for the permission name would
# still find it. Only the probe_write marker distinguishes them.
pw_sabotage "actions:write"
want_rc_msg 1 "actions:write is written by a call that swallows" \
  "control: losing the actions:write probe is caught" \
  python3 "$TMP/pw/scripts/assert-preflight-covers-writes.py"

# This one also proves the continuation-joining works: the bot's PUT spans five
# lines and its `||` sits on the last. Matched line-by-line it would read as
# unsuppressed, and the guard would pass by construction.
pw_sabotage "contents:write"
want_rc_msg 1 "contents:write is written by a call that swallows" \
  "control: losing the contents:write probe is caught (multi-line call)" \
  python3 "$TMP/pw/scripts/assert-preflight-covers-writes.py"

# ---------------------------------------------------------------------------
# The preflight's verdict must be the verdict
# ---------------------------------------------------------------------------
# This workflow is schedule- and dispatch-only. It is not a PR check, it cannot
# be a required context, and a red run in the Actions tab notifies nobody by
# default -- so the check run it mints on the default branch's head is the only
# place its answer reaches a human. That check run used to be POSTed
# `conclusion=success` as the checks:write probe, before any verdict existed.
# Run 33807779248 is the receipt: it failed on contents:write at 21:24:43Z and
# left a green "Certification preflight / App permissions verified" on
# de42fb155e, minted half a second earlier by its own probe. Nobody looked
# further, and the certificate shipped placeholder certifiers for two more days.
mkdir -p "$TMP/pf/bin"
python3 - > "$TMP/pf/step.sh" <<'PFX'
import yaml, pathlib
d = yaml.safe_load(pathlib.Path(".github/workflows/certification-preflight.yml").read_text())
for st in d["jobs"]["preflight"]["steps"]:
    if (st.get("name") or "").startswith("Probe every"):
        print(st["run"]); break
PFX
pf_stub() {  # pf_stub <"deny" to refuse the contents:write PUT>
  : > "$TMP/pf/calls"
  cat > "$TMP/pf/bin/gh" <<STUB
#!/bin/bash
printf 'ARGV: %s\n' "\$*" >> "\$PFCALLS"
case "\$*" in
  *"-X PUT"*contents*)    [ "$1" = deny ] && exit 1 ;;
  *"-X POST"*check-runs*) echo 424242 ;;
esac
exit 0
STUB
  chmod +x "$TMP/pf/bin/gh"
}
pf_run() { ( PATH="$TMP/pf/bin:$PATH" PFCALLS="$TMP/pf/calls" GH_TOKEN=t REPO=o/r SHA=deadbeef \
             bash -e "$TMP/pf/step.sh" >/dev/null 2>&1 ); }
pf_verdict() { grep -o 'conclusion=[a-z]*' "$TMP/pf/calls" | tail -1; }

if [ -s "$TMP/pf/step.sh" ]; then
  # The mark must be minted UNRESOLVED. A check posted already-green cannot be
  # made to say anything else, whatever the probes then find.
  pf_stub allow; pf_run
  grep -q 'POST repos/o/r/check-runs .*status=in_progress' "$TMP/pf/calls" \
    && ok "the preflight's check run is minted in progress, not pre-concluded" \
    || bad "the preflight mints its check run already concluded -- the verdict cannot change it"
  [ "$(pf_verdict)" = "conclusion=success" ] \
    && ok "all probes green -> the check run concludes success" \
    || bad "all probes green -> the check run concluded '$(pf_verdict)'"

  # THE REAL CASE: contents:write denied, exactly as production is today.
  pf_stub deny; pf_run
  [ "$(pf_verdict)" = "conclusion=failure" ] \
    && ok "a denied contents:write -> the check run concludes FAILURE" \
    || bad "a denied contents:write left the check run at '$(pf_verdict)' -- the green lie is back"
  # The summary is multi-line, so it lands on its own lines of the call log --
  # the report reaches the API or it does not appear here at all.
  grep -q '^  FAIL  contents:write' "$TMP/pf/calls" \
    && ok "and the check run names the permission that was refused" \
    || bad "the check run reports a failure without saying which permission"

  # CONTROL: restore the pre-fix probe -- mint the check already-concluded and
  # never patch it -- in a COPY, and confirm the green lie reappears. Without
  # this the four checks above could be passing on a step that mints nothing.
  sed -e 's/-f status=in_progress/-f status=completed -f conclusion=success/' \
      -e '/gh api -X PATCH "repos\/\$REPO\/check-runs\/\$check_run_id"/,+4d' \
      "$TMP/pf/step.sh" > "$TMP/pf/step-sab.sh"
  ( PATH="$TMP/pf/bin:$PATH" PFCALLS="$TMP/pf/calls" GH_TOKEN=t REPO=o/r SHA=deadbeef \
    bash -e "$TMP/pf/step-sab.sh" >/dev/null 2>&1 )
  [ "$(pf_verdict)" = "conclusion=success" ] \
    && ok "control: pre-concluding the mark publishes success on a failing preflight" \
    || bad "control: the sabotage did not reproduce the green lie -- the checks above prove nothing"
else
  bad "setup: could not extract the preflight's probe step"
fi

# ---------------------------------------------------------------------------
# /stamp must not report success when it did not release the lane
# ---------------------------------------------------------------------------
python3 - > "$TMP/stamp.sh" <<'PY'
import yaml, pathlib
d = yaml.safe_load(pathlib.Path(".github/workflows/certification-commands.yml").read_text())
for st in d["jobs"]["command"]["steps"]:
    if st.get("name") == "/stamp and /seal":
        print(st["run"]); break
PY
if [ -s "$TMP/stamp.sh" ]; then
  mkdir -p "$TMP/sbin"
  cat > "$TMP/sbin/gh" <<'STUB'
#!/usr/bin/env bash
all="$*"
case "$all" in
  *"actions/runs/"*"/rerun"*) printf 'RERUN %s\n' "$all" >> "$SCALLS"; echo "gh: Resource not accessible by integration (HTTP 403)" >&2; exit 1 ;;
  # The merged lookup fetches JSON and pipes it through jq ITSELF (so an
  # unanswered API is distinguishable from "no run"), and jq reads `.status`
  # (so an in-flight run is not re-run). The stub therefore has to be JSON,
  # like #908's, AND carry a status, like main's. `RUNMETA` stays the knob the
  # in-flight control turns -- "<id> <status>" -- so that control is unchanged.
  *"actions/runs?head_sha"*)
    _rm="${RUNMETA:-12345 completed}"
    printf '{"workflow_runs":[{"id":%s,"name":"CI","created_at":"2026-01-01T00:00:00Z","status":"%s"}]}\n' \
      "${_rm%% *}" "${_rm##* }"
    exit 0 ;;
  *"/jobs?per_page"*)
    # Which of the in-flight run's jobs have ALREADY finished. RUNJOBS is a
    # comma list of completed job names; empty = the run has not reached them.
    printf '{"jobs":['
    _first=1; _saved_ifs=$IFS; IFS=','
    for _j in ${RUNJOBS:-}; do
      [ -z "$_j" ] && continue
      [ "$_first" -eq 1 ] || printf ','
      printf '{"name":"%s","status":"completed"}' "$_j"; _first=0
    done
    IFS=$_saved_ifs; printf ']}\n'
    exit 0 ;;
  *"issues/"*"/comments"*)    printf '%s\n' "$all" >> "$SCALLS"; exit 0 ;;
  *)                          exit 0 ;;
esac
STUB
  chmod +x "$TMP/sbin/gh"
  : > "$TMP/scalls"
  ( PATH="$TMP/sbin:$PATH" SCALLS="$TMP/scalls" REPO=o/r PR=1 VERB=/stamp ACTOR=me AUTHOR=me \
    PERM=write HEAD_SHA=abc1234567 SHORT=abc1234 bash "$TMP/stamp.sh" >/dev/null 2>&1 )
  src=$?
  if [ "$src" -ne 0 ]; then
    ok "control: a stamp that could not release the lane fails the command job"
  else
    bad "control: the stamp step reported success while the lane stayed held"
  fi
  if grep -q "but CI was not re-run" "$TMP/scalls"; then
    ok "control: the PR comment says CI was not re-run"
  else
    bad "control: the comment claimed a clean stamp after the re-run failed"
  fi
  if grep -q "403" "$TMP/scalls"; then
    ok "control: the comment carries what the API actually said"
  else
    bad "control: the API's reason was discarded again"
  fi

  # THE LISTING, not just the re-run. `run_id=$(gh api ... 2>/dev/null || true)`
  # made an unanswered API indistinguishable from "this sha has no CI run": the
  # benign branch said "nothing to re-run", $rerun_note stayed empty, the comment
  # read "**Stamp recorded.** ... This holds for the life of the PR." and the job
  # went green -- while the held lane kept its `skipped` result and the gate went
  # on telling the operator to comment /stamp. Same defect as #847, one call
  # earlier, and the checks above could not see it because their stub answers.
  s_run() {  # s_run <gh stub body>  -> echoes the step's exit code
    cat > "$TMP/sbin/gh" <<STUB
#!/usr/bin/env bash
all="\$*"
$1
STUB
    chmod +x "$TMP/sbin/gh"
    : > "$TMP/scalls"
    ( PATH="$TMP/sbin:$PATH" SCALLS="$TMP/scalls" REPO=o/r PR=1 VERB=/stamp ACTOR=me AUTHOR=me \
      PERM=write HEAD_SHA=abc1234567 SHORT=abc1234 bash "$TMP/stamp.sh" >/dev/null 2>&1 )
    echo $?
  }

  rc=$(s_run 'case "$all" in
  *"actions/runs?head_sha"*) echo "gh: Bad credentials (HTTP 401)" >&2; exit 1 ;;
  *"issues/"*"/comments"*)   printf "%s\n" "$all" >> "$SCALLS"; exit 0 ;;
  *)                         exit 0 ;;
esac')
  [ "$rc" -ne 0 ] \
    && ok "a stamp whose CI run could not be LOOKED UP fails the command job" \
    || bad "the stamp reported success after the run listing failed -- the lane is still held"
  grep -q "but CI was not re-run" "$TMP/scalls" \
    && ok "and the comment says CI was not re-run" \
    || bad "the comment claimed a clean stamp after the run listing failed"
  grep -q "401" "$TMP/scalls" \
    && ok "and it carries what the API actually said" \
    || bad "the listing failure's reason was discarded"

  # CONTROL, the other side: an API that ANSWERS "no CI run for this sha" is a
  # different fact and must stay benign, or every stamp before CI starts would
  # refuse. Absent and unanswered must not collapse into one branch.
  rc=$(s_run 'case "$all" in
  *"actions/runs?head_sha"*) echo "{\"workflow_runs\":[]}"; exit 0 ;;
  *"issues/"*"/comments"*)   printf "%s\n" "$all" >> "$SCALLS"; exit 0 ;;
  *)                         exit 0 ;;
esac')
  [ "$rc" -eq 0 ] \
    && ok "control: a sha that genuinely has no CI run still stamps cleanly" \
    || bad "control: a PR stamped before CI started was refused"
  grep -q "but CI was not re-run" "$TMP/scalls" \
    && bad "control: it warned about a re-run that was never needed" \
    || ok "control: and says nothing about a re-run that was never needed"

  # s_run leaves its last stub behind; the /seal control below wants the
  # original one (a listing that answers, a re-run that 403s).
  s_run 'case "$all" in
  *"actions/runs/"*"/rerun"*) printf "RERUN %s\n" "$all" >> "$SCALLS"; echo "gh: Resource not accessible by integration (HTTP 403)" >&2; exit 1 ;;
  *"actions/runs?head_sha"*)  _rm="${RUNMETA:-12345 completed}"; printf "{\"workflow_runs\":[{\"id\":%s,\"name\":\"CI\",\"created_at\":\"2026-01-01T00:00:00Z\",\"status\":\"%s\"}]}\n" "${_rm%% *}" "${_rm##* }"; exit 0 ;;
  *"issues/"*"/comments"*)    printf "%s\n" "$all" >> "$SCALLS"; exit 0 ;;
  *)                          exit 0 ;;
esac' >/dev/null

  # CONTROL: a SEAL must re-run too. `seal status` reads the Seal check run,
  # and ci.yml has no `check_run` trigger -- so minting the mark alone left a
  # sealed PR reporting an unsealed `seal status` until someone pushed a
  # commit, which is the one thing that VOIDS a seal.
  : > "$TMP/scalls"
  ( PATH="$TMP/sbin:$PATH" SCALLS="$TMP/scalls" REPO=o/r PR=1 VERB=/seal ACTOR=me AUTHOR=me \
    PERM=write HEAD_SHA=abc1234567 SHORT=abc1234 bash "$TMP/stamp.sh" >/dev/null 2>&1 )
  if grep -q '^RERUN ' "$TMP/scalls"; then
    ok "control: /seal re-runs CI so seal status is re-evaluated"
  else
    bad "control: /seal minted its mark and never refreshed the job that reads it"
  fi

  # CONTROL: stamping while CI is still in flight must NOT warn. Re-running an
  # unfinished run returns 403 "This workflow is already running", which the
  # warning would report as a failure -- and it is not one: a run still in
  # flight reads the mark when its gate jobs execute. Found in production the
  # first time the warning printed the API's own words, which is what it is for.
  : > "$TMP/scalls"
  ( PATH="$TMP/sbin:$PATH" SCALLS="$TMP/scalls" RUNMETA="777 queued" REPO=o/r PR=1 VERB=/stamp \
    ACTOR=me AUTHOR=me PERM=write HEAD_SHA=abc1234567 SHORT=abc1234 \
    bash "$TMP/stamp.sh" >/dev/null 2>&1 )
  src=$?
  if [ "$src" -eq 0 ]; then
    ok "control: stamping mid-run is not an error"
  else
    bad "control: stamping while CI was queued failed the command job"
  fi
  if grep -q '^RERUN ' "$TMP/scalls"; then
    bad "control: it tried to re-run a run that had not finished"
  else
    ok "control: a run still in flight is not re-run"
  fi
  if grep -q "but CI was not re-run" "$TMP/scalls"; then
    bad "control: it warned about a benign in-flight run"
  else
    ok "control: and no false warning is posted"
  fi

  # CONTROL: "in flight" is TWO cases and only one is benign. When the jobs that
  # READ the mark already finished -- before the mark existed -- their verdict
  # is final for this run, and a job inside an in-progress run cannot be
  # re-run. Silence there is how #934/#935/#908 each ended 2026-09-06 with a
  # green `Seal` check run and a red `seal status`.
  #
  # This control installs its OWN stub: whichever stub ran last may not answer
  # the /jobs call, and then this would measure the benign branch and pass for
  # the wrong reason.
  cat > "$TMP/sbin/gh" <<'STUB2'
#!/usr/bin/env bash
all="$*"
case "$all" in
  *"actions/runs/"*"/rerun"*) printf 'RERUN %s\n' "$all" >> "$SCALLS"; exit 0 ;;
  # ★ JSON, NOT A BARE STRING. The merged lookup pipes this through jq
  # ITSELF so that an unanswered API is distinguishable from "no run"; a
  # plain "<id> <status>" makes jq fail, the step takes the could-not-list
  # branch, and the two controls below then measure the WRONG branch and
  # report the stale-jobs warning as missing. That is what they did the
  # first time #908 and #937 met.
  *"actions/runs?head_sha"*)
    _rm="${RUNMETA:-12345 completed}"
    printf '{"workflow_runs":[{"id":%s,"name":"CI","created_at":"2026-01-01T00:00:00Z","status":"%s"}]}\n' \
      "${_rm%% *}" "${_rm##* }"
    exit 0 ;;
  *"/jobs?per_page"*)
    printf '{"jobs":['
    _first=1; _saved_ifs=$IFS; IFS=','
    for _j in ${RUNJOBS:-}; do
      [ -z "$_j" ] && continue
      [ "$_first" -eq 1 ] || printf ','
      printf '{"name":"%s","status":"completed"}' "$_j"; _first=0
    done
    IFS=$_saved_ifs; printf ']}\n'; exit 0 ;;
  *"issues/"*"/comments"*)    printf '%s\n' "$all" >> "$SCALLS"; exit 0 ;;
  *)                          exit 0 ;;
esac
STUB2
  chmod +x "$TMP/sbin/gh"
  : > "$TMP/scalls"
  ( PATH="$TMP/sbin:$PATH" SCALLS="$TMP/scalls" RUNMETA="777 in_progress" \
    RUNJOBS="seal status,stamp status" REPO=o/r PR=1 VERB=/seal \
    ACTOR=me AUTHOR=me PERM=write HEAD_SHA=abc1234567 SHORT=abc1234 \
    bash "$TMP/stamp.sh" >/dev/null 2>&1 )
  if grep -q "the checks that read it have already run" "$TMP/scalls"; then
    ok "control: a mark minted after its gate jobs already ran is reported"
  else
    bad "control: gate jobs that already ran were treated as not-yet-run"
  fi
  if grep -q 'seal status' "$TMP/scalls"; then
    ok "control: and the warning NAMES the jobs whose verdict is already final"
  else
    bad "control: the warning did not say which jobs had already run"
  fi
  # And the benign case must stay silent: nothing finished yet.
  : > "$TMP/scalls"
  ( PATH="$TMP/sbin:$PATH" SCALLS="$TMP/scalls" RUNMETA="777 in_progress" RUNJOBS="" \
    REPO=o/r PR=1 VERB=/seal ACTOR=me AUTHOR=me PERM=write HEAD_SHA=abc1234567 SHORT=abc1234 \
    bash "$TMP/stamp.sh" >/dev/null 2>&1 )
  if grep -q "already run" "$TMP/scalls"; then
    bad "control: warned although no gate job had finished"
  else
    ok "control: an in-flight run that has not reached its gate jobs still warns nothing"
  fi
else
  bad "could not extract the /stamp step"
fi

# ---------------------------------------------------------------------------
# PR telemetry must not record an unreadable diff as an empty one
# ---------------------------------------------------------------------------
# The collector calls `pulls/N/files` once per PR, and that call can fail. It
# used to fall through to `paths='[]'` with no warning, and every consumer of
# the record reads an empty path list as a MEASUREMENT: no targets, no owners,
# no promotion debt, no collisions -- the most reassuring record this view can
# emit, manufactured out of an API error and posted to the tracking issue.
#
# The Rust half (gate/telemetry.rs) has tests for how the marker is HONOURED.
# Nothing tested that the shell half still EMITS it, and the shell half is the
# one with no other coverage, so this runs the real step against a gh stub.
echo "== pr telemetry marks an unreadable diff =="
python3 - > "$TMP/collect.sh" <<'TELPY'
import yaml, pathlib
d = yaml.safe_load(pathlib.Path(".github/workflows/pr-telemetry.yml").read_text())
for st in d["jobs"]["render"]["steps"]:
    if st.get("name", "").startswith("Collect open PRs"):
        print(st["run"]); break
TELPY
if [ -s "$TMP/collect.sh" ]; then
  mkdir -p "$TMP/tbin" "$TMP/trun"
  # #1's files come back; #2's call fails the way a 404 does -- non-zero exit
  # AND an error body on stdout, so a length check would read that body as a
  # filename. That shape is why the collector guards on the exit status.
  cat > "$TMP/tbin/gh" <<'TELSTUB'
#!/usr/bin/env bash
case "$*" in
  *"pulls?state=all"*)
    echo '[{"number":1,"title":"one","author":"a","draft":false,"merged":false},{"number":2,"title":"two","author":"b","draft":false,"merged":false}]' ;;
  *"pulls/1/files"*) printf 'kernels/gb10/x.cu\n' ;;
  *"pulls/2/files"*) echo '{"message":"Not Found","status":"404"}'; exit 1 ;;
  *) exit 1 ;;
esac
TELSTUB
  chmod +x "$TMP/tbin/gh"
  tel_run() {  # $1 = step script -> facts.json on stdout
    ( cd "$TMP/trun" && rm -f facts.json prs.json prs.ndjson enriched.ndjson
      PATH="$TMP/tbin:$PATH" GH_TOKEN=x REPO=o/r bash "$1" >/dev/null 2>&1 )
    jq -c '.' "$TMP/trun/facts.json" 2>/dev/null
  }
  tel_run "$TMP/collect.sh" > "$TMP/facts.txt"
  if jq -e 'any(.[]; .number == 2 and .paths_unknown == true)' "$TMP/facts.txt" >/dev/null 2>&1; then
    ok "a PR whose files could not be read is recorded as paths_unknown"
  else
    bad "an API failure was recorded as a PR that changes nothing"
    sed 's/^/       /' "$TMP/facts.txt" | head -2
  fi
  # The marker must DISCRIMINATE. A collector that stamped every record
  # `paths_unknown: true` would pass the check above and make the view useless,
  # so the readable PR must come back marked known, with its path.
  if jq -e 'any(.[]; .number == 1 and .paths_unknown == false and (.changed_paths | length) == 1)' \
       "$TMP/facts.txt" >/dev/null 2>&1; then
    ok "a PR whose files WERE read is recorded as known"
  else
    bad "the marker does not discriminate: a readable diff came back unknown"
    sed 's/^/       /' "$TMP/facts.txt" | head -2
  fi
  # CONTROL: the pre-fix behaviour, reconstructed by flipping the one flag.
  # It must turn the first check red and leave the second green, which is what
  # proves the check measures the failure path and not the happy one.
  sed 's/^\( *\)unknown=true$/\1unknown=false/' "$TMP/collect.sh" > "$TMP/collect-bad.sh"
  if cmp -s "$TMP/collect.sh" "$TMP/collect-bad.sh"; then
    bad "control: the sabotage did not change the step -- it would measure nothing"
  else
    tel_run "$TMP/collect-bad.sh" > "$TMP/facts-bad.txt"
    if jq -e 'any(.[]; .number == 2 and .paths_unknown == false)' "$TMP/facts-bad.txt" >/dev/null 2>&1 \
       && jq -e 'any(.[]; .number == 1 and .paths_unknown == false)' "$TMP/facts-bad.txt" >/dev/null 2>&1; then
      ok "control: dropping the marker is caught (#2 renders as changing nothing)"
    else
      bad "control: the sabotaged collector did not reproduce the defect"
      sed 's/^/       /' "$TMP/facts-bad.txt" | head -2
    fi
  fi
else
  bad "could not extract the telemetry collect step"
fi

# ---------------------------------------------------------------------------
# The PR classifier must never run on a diff that could not be computed
# ---------------------------------------------------------------------------
# `pr-categorize` builds the model's context from the changed paths, and the
# ★ note on that step records why: when the paths were computed and never
# read, "the classifier's entire input was attacker-authored prose -- titling
# a decode-kernel PR 'docs: tidy a comment' was a complete bypass".
#
# The collection step reopened that bypass from the other end. `git diff ... >
# changed.txt || true` truncates the file BEFORE git runs, so a failed diff
# leaves it empty and the prompt reads "Changed paths (0 total)" -- a PR that
# changes nothing, classified from its title and body, with the model's answer
# appended to the journey ledger as evidence. The deterministic fallback
# abstains too (`all_match` needs total > 0), so nothing catches it.
echo "== the pr classifier refuses an uncomputable diff =="
python3 - > "$TMP/diffstep.sh" <<'DFPY'
import yaml, pathlib
d = yaml.safe_load(pathlib.Path(".github/workflows/ci.yml").read_text())
for st in d["jobs"]["pr-categorize"]["steps"]:
    if st.get("name") == "Collect the changed paths":
        print(st["run"]); break
DFPY
if [ -s "$TMP/diffstep.sh" ]; then
  mkdir -p "$TMP/gbin" "$TMP/grun"
  cat > "$TMP/gbin/git" <<'GITSTUB'
#!/usr/bin/env bash
# `diff` answers per GIT_STUB_MODE; anything else is not this step's business.
case "$1" in
  diff)
    case "${GIT_STUB_MODE:-ok}" in
      # The real failure: a base sha that is no longer reachable. git writes
      # nothing to stdout, so the redirect leaves an empty file behind.
      fail) echo "fatal: bad object $2" >&2; exit 128 ;;
      *)    printf 'kernels/gb10/x.cu\ncrates/spark-server/src/lib.rs\n' ;;
    esac ;;
  *) exit 0 ;;
esac
GITSTUB
  chmod +x "$TMP/gbin/git"
  df_run() {  # $1 = step script, $2 = stub mode -> rc; step outputs in $TMP/gout
    : > "$TMP/gout"
    ( cd "$TMP/grun" && rm -f changed.txt histogram.txt sorted.txt paths.txt
      PATH="$TMP/gbin:$PATH" GITHUB_OUTPUT="$TMP/gout" GIT_STUB_MODE="$2" \
        BASE_SHA=aaaaaaa HEAD_SHA=bbbbbbb bash "$1" >"$TMP/gerr" 2>&1 )
  }
  df_run "$TMP/diffstep.sh" ok
  if [ $? -eq 0 ] && grep -q '^count=2$' "$TMP/gout"; then
    ok "a diff that CAN be computed is emitted with its real count"
  else
    bad "the step no longer works on a readable diff"
    sed 's/^/       /' "$TMP/gerr" | head -3
  fi
  df_run "$TMP/diffstep.sh" fail
  drc=$?
  if [ "$drc" -ne 0 ]; then
    ok "a diff that cannot be computed fails the step"
  else
    bad "a failed git diff was recorded as a PR that changes nothing ($(grep -m1 '^count=' "$TMP/gout"))"
  fi
  # The reason has to reach the run, not just the exit code: this job is
  # advisory, so a bare non-zero with no annotation is a red X nobody can act
  # on -- which is how it gets rerun-until-green instead of fixed.
  if grep -q '::error title=Changed paths unavailable' "$TMP/gerr"; then
    ok "the failure names itself in the run's annotations"
  else
    bad "the step failed silently; no annotation says why"
  fi
  # CONTROL: put the `|| true` back and watch the failure become "0 paths".
  sed 's|^\( *\)if ! git diff --name-only "\$BASE_SHA" "\$HEAD_SHA" > changed.txt; then|\1git diff --name-only "$BASE_SHA" "$HEAD_SHA" > changed.txt \|\| true\n\1if false; then|' \
    "$TMP/diffstep.sh" > "$TMP/diffstep-bad.sh"
  if cmp -s "$TMP/diffstep.sh" "$TMP/diffstep-bad.sh"; then
    bad "control: the sabotage did not change the step -- it would measure nothing"
  else
    df_run "$TMP/diffstep-bad.sh" fail
    if [ $? -eq 0 ] && grep -q '^count=0$' "$TMP/gout"; then
      ok "control: with '|| true' restored, a failed diff emits count=0"
    else
      bad "control: the sabotage did not reproduce the fail-open defect"
      sed 's/^/       /' "$TMP/gerr" | head -3
    fi
  fi
else
  bad "could not extract the changed-paths step"
fi

# ---------------------------------------------------------------------------
# The unsigned-record guard must not pass on a diff it could not read
# ---------------------------------------------------------------------------
# "One PR, one commit, one signer" is the check that stops a PR adding an
# UNSIGNED benchmark record -- the ci.yml comment beside it records the
# experiment: an unsigned record dated 1700000000 claiming 999 tok/s passes
# the Rust gate, so this shell step is the thing standing in its way.
#
# Its entire input was `mapfile -t added < <(git diff ... | grep ...)`, and a
# process substitution's exit status is invisible to `set -euo pipefail`. A
# failed enumeration therefore produced an empty array, and empty is the one
# value this guard reads as "nothing to check": it printed "this PR adds no
# records" and exited 0, waving through every unsigned record, every
# cross-commit set and every second signer at once.
echo "== the unsigned-record guard reads its own input =="
python3 - > "$TMP/signer.sh" <<'SGPY'
import yaml, pathlib
d = yaml.safe_load(pathlib.Path(".github/workflows/ci.yml").read_text())
for st in d["jobs"]["pr-benchmark-gate"]["steps"]:
    if st.get("name") == "One PR, one commit, one signer":
        print(st["run"]); break
SGPY
if [ -s "$TMP/signer.sh" ]; then
  mkdir -p "$TMP/sgbin" "$TMP/sgrun/.benchmarks/x"
  # An added record with NO .json.sig beside it -- the case the guard exists
  # for. It must be caught when the diff works, and must NOT be silently
  # excused when the diff does not.
  echo '{"git_sha":"abc1234567"}' > "$TMP/sgrun/.benchmarks/x/2026-01-01-abc1234567.json"
  cat > "$TMP/sgbin/git" <<'SGSTUB'
#!/usr/bin/env bash
case "$1" in
  merge-base) echo base0000000 ;;
  diff)
    case "${GIT_STUB_MODE:-ok}" in
      fail) echo "fatal: bad revision" >&2; exit 128 ;;
      *)    echo ".benchmarks/x/2026-01-01-abc1234567.json" ;;
    esac ;;
  *) exit 0 ;;
esac
SGSTUB
  chmod +x "$TMP/sgbin/git"
  sg_run() {  # $1 = step script, $2 = stub mode -> rc, output in $TMP/sgout
    ( cd "$TMP/sgrun" && rm -f added_all.txt
      PATH="$TMP/sgbin:$PATH" GIT_STUB_MODE="$2" BASE=deadbeef \
        bash "$1" >"$TMP/sgout" 2>&1 )
  }
  sg_run "$TMP/signer.sh" ok
  if [ $? -ne 0 ] && grep -q 'Unsigned record added' "$TMP/sgout"; then
    ok "an added record with no sidecar is refused"
  else
    bad "the guard stopped catching an unsigned record"
    sed 's/^/       /' "$TMP/sgout" | head -3
  fi
  sg_run "$TMP/signer.sh" fail
  sgrc=$?
  if [ "$sgrc" -ne 0 ] && ! grep -q 'adds no records' "$TMP/sgout"; then
    ok "an unreadable diff fails the guard instead of clearing the PR"
  else
    bad "a failed enumeration was reported as a PR that adds no records"
    sed 's/^/       /' "$TMP/sgout" | head -3
  fi
  if grep -q 'It is NOT reporting that the PR adds none' "$TMP/sgout"; then
    ok "the annotation distinguishes 'could not look' from 'found nothing'"
  else
    bad "the guard failed without saying it had been blinded"
  fi
  # CONTROL: the pre-fix single line, restored. The unsigned record must go
  # through, which is the defect this guard is here to make impossible.
  python3 - "$TMP/signer.sh" "$TMP/signer-bad.sh" <<'SGSAB'
import pathlib, sys, re
t = pathlib.Path(sys.argv[1]).read_text()
new = re.sub(
    r'if ! git diff --name-only --diff-filter=AM "\$base"\.\.\.HEAD -- \.benchmarks \\\n'
    r' *> added_all\.txt; then\n.*?\n *exit 1\n *fi\n',
    '', t, count=1, flags=re.S)
new = new.replace(
    "mapfile -t added < <(grep '\\.json$' added_all.txt || true)",
    "mapfile -t added < <(git diff --name-only --diff-filter=AM \"$base\"...HEAD "
    "-- .benchmarks | grep '\\.json$' || true)", 1)
assert new != t, "sabotage did not change the step -- it would measure nothing"
pathlib.Path(sys.argv[2]).write_text(new)
SGSAB
  if [ -s "$TMP/signer-bad.sh" ]; then
    sg_run "$TMP/signer-bad.sh" fail
    if [ $? -eq 0 ] && grep -q 'adds no records' "$TMP/sgout"; then
      ok "control: the old form clears an unsigned record when the diff fails"
    else
      bad "control: the sabotage did not reproduce the fail-open defect"
      sed 's/^/       /' "$TMP/sgout" | head -3
    fi
  else
    bad "control: could not build the sabotaged step"
  fi
else
  bad "could not extract the one-signer step"
fi

# A required check that cannot vouch for the tree must not say OK
# ---------------------------------------------------------------------------
# "No block_on under tui/ or recipe/" is one of main's required contexts, and it
# is one grep. grep exits 2 when the SCAN fails -- a renamed or deleted
# directory, an unreadable file -- and GNU grep returns 2 even when it also
# found matches. Written as `if hits=$(grep ... 2>/dev/null); then`, exit 2 is
# "false", so the step fell through to `echo "OK: ..."` and went green for a
# scan that never happened. The two directory names are hard-coded again in the
# workflow's push `paths:` filter, so renaming one is exactly the change that
# would have hit it -- and it would have taken the gate with it, silently.
mkdir -p "$TMP/tt"
python3 - > "$TMP/tt/step.sh" <<'TTX'
import yaml, pathlib
d = yaml.safe_load(pathlib.Path(".github/workflows/tui-threading.yml").read_text())
for st in d["jobs"]["no-blocking-on-the-render-thread"]["steps"]:
    if "run" in st:
        print(st["run"]); break
TTX
tt_tree() {  # tt_tree <clean|dirty|renamed>
  rm -rf "$TMP/tt/w"; mkdir -p "$TMP/tt/w/crates/spark-server/src/tui"
  case "$1" in
    renamed) mkdir -p "$TMP/tt/w/crates/spark-server/src/recipes" ;;
    *)       mkdir -p "$TMP/tt/w/crates/spark-server/src/recipe" ;;
  esac
  printf 'fn tick() { let _ = rx.try_recv(); }\n' > "$TMP/tt/w/crates/spark-server/src/tui/chat.rs"
  [ "$1" = dirty ] && printf 'fn tick() { rt.block_on(f); }\n' > "$TMP/tt/w/crates/spark-server/src/tui/bad.rs"
  return 0
}
# SCAN_DIRS comes from the job's `env:` in the workflow, so the extracted step
# dies on `set -u` without it. Read it OUT OF the workflow rather than
# repeating the value: the point of SCAN_DIRS is that the existence assertion
# and the grep cannot disagree about which trees they cover, and a hard-coded
# copy here would be a third opinion able to drift from both.
TT_SCAN_DIRS=$(python3 - <<'SDPY'
import yaml
d = yaml.safe_load(open(".github/workflows/tui-threading.yml"))
jid = list(d["jobs"])[0]
for st in d["jobs"][jid]["steps"]:
    if st.get("env", {}).get("SCAN_DIRS"):
        print(st["env"]["SCAN_DIRS"]); break
SDPY
)
[ -n "$TT_SCAN_DIRS" ] || bad "setup: could not read SCAN_DIRS out of tui-threading.yml"
tt_run() { ( cd "$TMP/tt/w" && SCAN_DIRS="$TT_SCAN_DIRS" bash "$1" ) >"$TMP/tt/out" 2>&1; echo $?; }

if [ -s "$TMP/tt/step.sh" ]; then
  tt_tree clean
  [ "$(tt_run "$TMP/tt/step.sh")" = 0 ] && grep -q '^OK: ' "$TMP/tt/out" \
    && ok "a clean tree passes the render-thread check" \
    || bad "a clean tree did not pass: $(head -2 "$TMP/tt/out")"
  tt_tree dirty
  [ "$(tt_run "$TMP/tt/step.sh")" != 0 ] \
    && ok "control: a real block_on under tui/ is caught" \
    || bad "control: a block_on under tui/ passed"
  # THE ONE THAT SHIPPED: the scan could not read the tree.
  tt_tree renamed
  if [ "$(tt_run "$TMP/tt/step.sh")" != 0 ]; then
    ok "a scan that could not read the tree fails, instead of reporting OK"
  else
    bad "a renamed directory made this required check report 'OK' on a tree it never read"
  fi
  grep -q '^OK: ' "$TMP/tt/out" \
    && bad "and it printed OK for a scan that did not run" \
    || ok "and it does not print OK for a scan that did not run"

  # CONTROL: the pre-fix form -- `if hits=$(grep ... 2>/dev/null)` -- restored in
  # a COPY, on the same unreadable tree. It must go green, which is what proves
  # the two checks above are not passing by construction.
  cat > "$TMP/tt/step-sab.sh" <<'SAB'
set -euo pipefail
if hits=$(grep -rnE '\.(block_on|block_in_place)\(' \
            --include='*.rs' --exclude='*_tests.rs' \
            crates/spark-server/src/tui/ \
            crates/spark-server/src/recipe/ 2>/dev/null); then
  echo "::error::The TUI render thread must never poll a future."
  echo "$hits"
  exit 1
fi
echo "OK: no block_on/block_in_place under tui/ or recipe/"
SAB
  tt_tree renamed
  [ "$(tt_run "$TMP/tt/step-sab.sh")" = 0 ] && grep -q '^OK: ' "$TMP/tt/out" \
    && ok "control: the pre-fix form reports OK on a tree it could not read" \
    || bad "control: the sabotage did not reproduce the defect -- the checks above prove nothing"
else
  bad "setup: could not extract the render-thread check's step"
fi

# ---------------------------------------------------------------------------
# The install canary must not report a probe it could not make
# ---------------------------------------------------------------------------
# `code=$(curl ... --write-out '%{http_code}' ... || echo 000)`. curl prints
# `000` on a transport failure ALREADY -- --write-out runs either way -- so the
# `|| echo 000` appended a second one and `code` became the two-line string
# "000\n000". That matches neither `5*` nor `000`, so the case fell through to
# `ok   /control -> 000000 (fallback, not an error)` and the nightly canary
# called a timed-out or TLS-failed /control healthy. This probe has no --retry,
# unlike the `check` helper above it, so it is the likeliest one to trip.
mkdir -p "$TMP/ic/bin"
python3 - > "$TMP/ic/step.sh" <<'ICX'
import yaml, pathlib
d = yaml.safe_load(pathlib.Path(".github/workflows/install-canary.yml").read_text())
for st in d["jobs"]["pages"]["steps"]:
    if "run" in st:
        print(st["run"]); break
ICX
# A curl that serves a healthy site, except that /control can be made to fail at
# the transport level -- printing 000 and exiting non-zero, exactly as curl does.
ic_curl() {  # ic_curl <"break" to fail the /control probe>
  cat > "$TMP/ic/bin/curl" <<STUB
#!/bin/bash
url=""; for a in "\$@"; do case "\$a" in https://*) url="\$a" ;; esac; done
case "\$url" in
  */control)      [ "$1" = break ] && { printf '000'; exit 7; }; printf '200'; exit 0 ;;
  */control.html) printf '<title>Control plane</title>' ;;
  */install.sh)   printf '#!/bin/sh\nexit 0\n' ;;
  */install.ps1)  printf '# atlas installer\n' ;;
  *)              printf '<title>Atlas, pure Rust inference</title>' ;;
esac
exit 0
STUB
  chmod +x "$TMP/ic/bin/curl"
}
ic_run() { ( PATH="$TMP/ic/bin:$PATH" bash "$1" ) >"$TMP/ic/out" 2>&1; echo $?; }

if [ -s "$TMP/ic/step.sh" ]; then
  ic_curl healthy
  [ "$(ic_run "$TMP/ic/step.sh")" = 0 ] \
    && ok "a healthy site passes the install canary" \
    || bad "a healthy site failed the canary: $(grep -m1 error "$TMP/ic/out")"

  ic_curl break
  if [ "$(ic_run "$TMP/ic/step.sh")" != 0 ]; then
    ok "a /control probe that could not be made fails the canary"
  else
    bad "a /control probe that never completed was reported as healthy"
  fi
  grep -q 'fallback, not an error' "$TMP/ic/out" \
    && bad "and it called the unmade probe 'ok ... (fallback, not an error)'" \
    || ok "and it does not call the unmade probe a fallback"

  # CONTROL: the pre-fix `|| echo 000` restored in a COPY, same broken curl.
  python3 - "$TMP/ic/step.sh" "$TMP/ic/step-sab.sh" <<'ICSAB'
import pathlib, sys
t = pathlib.Path(sys.argv[1]).read_text()
old = "--location --max-time 20 https://atlasinference.io/control) || true"
new = "--location --max-time 20 https://atlasinference.io/control || echo 000)"
pathlib.Path(sys.argv[2]).write_text(t.replace(old, new, 1))
ICSAB
  if ! cmp -s "$TMP/ic/step.sh" "$TMP/ic/step-sab.sh"; then
    ic_curl break
    [ "$(ic_run "$TMP/ic/step-sab.sh")" = 0 ] && grep -q 'fallback, not an error' "$TMP/ic/out" \
      && ok "control: the pre-fix form calls a failed /control probe healthy" \
      || bad "control: the sabotage did not reproduce the defect -- the checks above prove nothing"
  else
    bad "control setup: the sabotage did not land; the probe's shape has changed"
  fi
else
  bad "setup: could not extract the install canary's page step"
fi

# ---------------------------------------------------------------------------
# nginx add_header does not accumulate across contexts
# ---------------------------------------------------------------------------
# Three vhosts, one rule, three separate incidents -- each found by hand after
# the fact. The docs vhost served every HTML document with no security headers
# because Cache-Control sat inside `location ~* \.html$`; the site vhost
# discarded Alt-Svc on every proxied response and sent its dotfile refusal bare;
# and the site vhost was independently missing Referrer-Policy that the other
# two sent. All three are fixed. Nothing stopped a fourth.
#
# There is no historical config to replay: the docs vhost has a single commit,
# so the pre-fix text is not in the tree. The sabotages below reconstruct the
# defect SHAPE instead, which is stated plainly rather than dressed up as a
# regression test against real history.
want_rc 0 "the three vhosts agree, at server level" \
  python3 .github/scripts/assert-vhost-headers.py
mkdir -p "$TMP/vh/.github/scripts"
cp .github/scripts/assert-vhost-headers.py "$TMP/vh/.github/scripts/"

vh_sabotage() {  # $1 = vhost path, $2 = python edit over the file text as `t`
  for f in site/deploy/nginx/atlasinference.io.conf \
           blog/deploy/nginx/blog.atlasinference.io.conf \
           book/deploy/nginx/docs.atlasinference.io.conf; do
    mkdir -p "$TMP/vh/$(dirname "$f")"; cp "$f" "$TMP/vh/$f"
  done
  [ -n "${1:-}" ] || return 0
  python3 - "$TMP/vh/$1" "$2" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text()
ns = {"t": t}; exec(sys.argv[2], ns)
assert ns["t"] != t, "sabotage did not change the file -- the control would measure nothing"
p.write_text(ns["t"])
PY
}

# The docs incident, reconstructed: a location that declares any add_header
# drops every inherited one for that path.
vh_sabotage book/deploy/nginx/docs.atlasinference.io.conf \
  't = t.replace("    location / {", "    location ~* \\\\.html$ {\n        add_header Cache-Control \"no-store\" always;\n    }\n\n    location / {", 1)'
want_rc_msg 1 "inside a location" "control: an add_header inside a location is caught" \
  python3 "$TMP/vh/.github/scripts/assert-vhost-headers.py"

# The site incident: one vhost quietly lacking a header the other two send.
vh_sabotage blog/deploy/nginx/blog.atlasinference.io.conf \
  't = t.replace("    add_header Referrer-Policy", "    # add_header Referrer-Policy", 1)'
want_rc_msg 1 "does not declare Referrer-Policy" "control: a vhost missing a core header is caught" \
  python3 "$TMP/vh/.github/scripts/assert-vhost-headers.py"

vh_sabotage blog/deploy/nginx/blog.atlasinference.io.conf \
  't = t.replace("    add_header Referrer-Policy", "    # add_header Referrer-Policy", 1)'
want_rc_msg 1 "have drifted" "control: drift between the vhosts is named as drift" \
  python3 "$TMP/vh/.github/scripts/assert-vhost-headers.py"

# The value that was live on atlasinference.io when this was written.
vh_sabotage site/deploy/nginx/atlasinference.io.conf \
  't = t.replace("X-XSS-Protection \"0\"", "X-XSS-Protection \"1; mode=block\"", 1)'
want_rc_msg 1 "OWASP" "control: re-enabling the legacy XSS auditor is caught" \
  python3 "$TMP/vh/.github/scripts/assert-vhost-headers.py"

# ---------------------------------------------------------------------------
# Markdown links that point at nothing
# ---------------------------------------------------------------------------
# docs/lora-implementation-status.md linked to two files that have never existed
# in this repository's history -- dead the day it was committed.
#
# The controls below matter more than usual because the throwaway sweep that
# found that defect reported nineteen broken links and SEVENTEEN were its own
# bugs (it stripped the dot from `.github`, and resolved `/images/...` against
# the filesystem). A checker that cries wolf gets muted. So the controls prove
# both directions: that real breakage is caught, AND that the site-root and
# generated-path cases are resolved rather than quietly skipped -- a checker
# that skips what it cannot resolve passes vacuously.
want_rc 0 "every in-repo markdown link resolves" \
  python3 .github/scripts/assert-doc-links.py

dl_tree() {  # build a miniature repo the checker can be pointed at
  rm -rf "$TMP/dl"
  mkdir -p "$TMP/dl/.github/scripts" "$TMP/dl/docs" \
           "$TMP/dl/blog/src" "$TMP/dl/blog/static/images" "$TMP/dl/book/src"
  cp .github/scripts/assert-doc-links.py "$TMP/dl/.github/scripts/"
  : > "$TMP/dl/docs/target.md"
  printf 'PNG' > "$TMP/dl/blog/static/images/hero.webp"
  printf '[ok](target.md)\n'          > "$TMP/dl/docs/good.md"
  printf '![h](/images/hero.webp)\n'  > "$TMP/dl/blog/src/post.md"
  printf '[api](/api/atlas_core/)\n'  > "$TMP/dl/book/src/redirect.md"
}
dl_run() { python3 "$TMP/dl/.github/scripts/assert-doc-links.py"; }

dl_tree
want_rc 0 "control: a good tree passes (relative, site-root and generated all resolve)" dl_run

# The defect exactly as it was found.
dl_tree; printf '[mvp](lora-mvp-proposal.md)\n' > "$TMP/dl/docs/dead.md"
want_rc_msg 1 "no such file" "control: a relative link to a missing file is caught" dl_run

# If site-root links were skipped rather than resolved, this would still pass.
dl_tree; rm "$TMP/dl/blog/static/images/hero.webp"
want_rc_msg 1 "no such file" "control: a site-root link is really resolved, not skipped" dl_run

# A site-root link from a tree that publishes no static dir must be loud, not
# silently ignored -- silence is how the seventeen false negatives would hide.
dl_tree; printf '![x](/images/hero.webp)\n' > "$TMP/dl/docs/rooted.md"
want_rc_msg 1 "no known static root" "control: a site-root link from an unknown tree is refused" dl_run

# And the generated rustdoc path must NOT be flagged: docs.yml assembles it at
# deploy time, so requiring it in the tree would fail every PR forever.
dl_tree; printf '[a](/api/spark_model/)\n[b](/api/)\n' > "$TMP/dl/book/src/more.md"
want_rc 0 "control: generated /api/ paths are not treated as breakage" dl_run

# ---------------------------------------------------------------------------
# Commands that exist only in a workflow file
# ---------------------------------------------------------------------------
# Three of the five -- /help, /review and /expedite -- were accepted by the bot
# and mentioned nowhere in the README. /expedite skips certification outright.
# All three arrived during this record's own waves, one commit at a time, and no
# single change looked like it was leaving something out.
want_rc 0 "every accepted command is documented" \
  python3 .github/scripts/assert-commands-documented.py
mkdir -p "$TMP/cd/.github/workflows"
cp .github/scripts/assert-commands-documented.py "$TMP/cd/.github/"
mkdir -p "$TMP/cd/.github/scripts"
mv "$TMP/cd/.github/assert-commands-documented.py" "$TMP/cd/.github/scripts/"

cd_case() { printf '%s\n' \
  "jobs:" "  command:" "    steps:" "      - run: |" "          case \"\$VERB\" in" \
  "            $1) ;;" "            *) exit 0 ;;" "          esac" \
  > "$TMP/cd/.github/workflows/certification-commands.yml"; }

cd_case '/help|/stamp|/seal'
printf 'we document /help and /stamp and /seal here.\n' > "$TMP/cd/README.md"
want_rc 0 "control: a README covering every verb passes" \
  python3 "$TMP/cd/.github/scripts/assert-commands-documented.py"

cd_case '/help|/stamp|/seal|/expedite'
want_rc_msg 1 "accepts \`/expedite\`" "control: a command missing from the README is caught" \
  python3 "$TMP/cd/.github/scripts/assert-commands-documented.py"

# If the case arm is renamed or restructured, the guard must REFUSE rather than
# find nothing and report success -- a guard that cannot locate its input and
# passes is the failure mode this whole suite exists to catch.
printf 'jobs:\n  command:\n    steps:\n      - run: echo no case arm here\n' \
  > "$TMP/cd/.github/workflows/certification-commands.yml"
want_rc_msg 1 "could not find the verb whitelist" \
  "control: a guard that cannot find its input refuses, it does not pass" \
  python3 "$TMP/cd/.github/scripts/assert-commands-documented.py"

echo
echo "  $PASS passed, $FAIL failed"
REACHED_SUMMARY=1
[ "$FAIL" -eq 0 ]
