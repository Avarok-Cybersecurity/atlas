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

echo "== one PR, one commit, one signer =="
extract 'pr-benchmark-gate/One PR, one commit, one signer' > "$TMP/oc.sh"
if [ -s "$TMP/oc.sh" ]; then
  R="$TMP/ocrepo"; rm -rf "$R"; mkdir -p "$R/.benchmarks/b"
  ( cd "$R" && git init -q . && git config user.email t@t && git config user.name t \
    && echo s > s && git add -A && git commit -qm base ) >/dev/null 2>&1
  BASE=$( cd "$R" && git rev-parse HEAD )
  mk() { printf '{"git_sha":"%s","recorded_at":1788300000}' "$2" > "$R/.benchmarks/b/$1.json"; }
  sg() { printf '{"v":1,"key":"%s","sig":"x"}' "$2" > "$R/.benchmarks/b/$1.json.sig"; }
  mk r1 aaaaaaaaaa; sg r1 k1; ( cd "$R" && git add -A && git commit -qm r1 ) >/dev/null 2>&1
  want_rc 0 "records that agree pass" sh -c "cd '$R' && BASE=$BASE bash '$TMP/oc.sh'"
  mk r2 bbbbbbbbbb; sg r2 k1; ( cd "$R" && git add -A && git commit -qm r2 ) >/dev/null 2>&1
  want_rc_msg 1 "span more than one commit" "control: records from two commits are refused" \
    sh -c "cd '$R' && BASE=$BASE bash '$TMP/oc.sh'"
  mk r2 aaaaaaaaaa; sg r2 k2; ( cd "$R" && git add -A && git commit -qm r3 ) >/dev/null 2>&1
  want_rc_msg 1 "more than one signer" "control: records from two signers are refused" \
    sh -c "cd '$R' && BASE=$BASE bash '$TMP/oc.sh'"
  # The backdating bypass: an ADDED record with no signature at all.
  rm -f "$R/.benchmarks/b/r2.json.sig"; sg r1 k1
  ( cd "$R" && git add -A && git rm -q --cached .benchmarks/b/r2.json.sig 2>/dev/null; git commit -qm r4 ) >/dev/null 2>&1
  want_rc_msg 1 "Unsigned record added" "control: an added record with no signature is refused" \
    sh -c "cd '$R' && BASE=$BASE bash '$TMP/oc.sh'"
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
  for w in site.yml merge-ancestry.yml; do cp ".github/workflows/$w" "$TMP/sg/workflows/$w"; done
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
  *"actions/runs?head_sha"*)  echo 12345; exit 0 ;;
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
else
  bad "could not extract the /stamp step"
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

echo
echo "  $PASS passed, $FAIL failed"
REACHED_SUMMARY=1
[ "$FAIL" -eq 0 ]
