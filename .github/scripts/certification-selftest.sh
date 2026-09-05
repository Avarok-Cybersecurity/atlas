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
    if grep -q 'certificate-merged.png' "$TMP/bcalls"; then
      ok "control: and it points at the generic image, not a render that never happened"
    else
      bad "control: it linked a rendered certificate that was never produced"
    fi
  fi

  runbot pr-certification-stage-1 "" 0
  posted && ! patched && ok "no prior comment -> posts one" || bad "no prior comment -> did not post exactly one"
  # CONTROL: with a prior comment it must EDIT, never post a second. A thread of
  # stale state comments is precisely what the marker exists to prevent.
  runbot pr-certification-stage-1 12345 0
  patched && ok "control: an existing comment is edited, not duplicated" \
    || bad "control: it posted a second state comment instead of editing"

  # The marker is both the lookup key and the memory of the previous state.
  grep -q 'atlas-certification-state' "$TMP/bot.sh" \
    && ok "the state marker is written into the comment" \
    || bad "no state marker -- the next run cannot find its own comment"

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
