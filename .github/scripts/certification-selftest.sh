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
# A test suite that gates nothing (#810)
# ---------------------------------------------------------------------------
# `Site unit tests` ran on every PR and blocked neither merge nor deploy. The
# guard below is pinned here rather than in site.yml's own lane on purpose: a
# suite cannot be trusted to notice that it has been unwired from itself.
#
# Each control pins the *message* as well as the exit code. Three different
# defects all exit 1 through the same `fail()`, so asserting rc alone would
# assert nothing about which one fired.
want_rc 0 "the site suite gates the deploy as things stand" \
  python3 .github/scripts/assert-site-tests-gate.py
mkdir -p "$TMP/sg/scripts" "$TMP/sg/workflows"
cp .github/scripts/assert-site-tests-gate.py "$TMP/sg/scripts/"

sg_sabotage() {  # $1 = python edit applied to the copied site.yml
  cp .github/workflows/site.yml "$TMP/sg/workflows/site.yml"
  python3 - "$TMP/sg/workflows/site.yml" "$1" <<'PY'
import pathlib, sys, yaml
p = pathlib.Path(sys.argv[1]); d = yaml.safe_load(p.read_text())
exec(sys.argv[2], {"d": d})
p.write_text(yaml.safe_dump(d, sort_keys=False))
PY
}

sg_sabotage 'd["jobs"]["deploy"]["needs"] = ["build"]'
want_rc_msg 1 "does not need" "control: dropping unit from deploy's needs is caught" \
  python3 "$TMP/sg/scripts/assert-site-tests-gate.py"

sg_sabotage 'd["jobs"]["unit"]["if"] = "github.event_name == \"push\""'
want_rc_msg 1 "grew an \`if:\`" "control: making the suite conditional is caught" \
  python3 "$TMP/sg/scripts/assert-site-tests-gate.py"

sg_sabotage 'd[True]["pull_request"] = {"branches": ["main"], "paths": ["site/**"]}'
want_rc_msg 1 "grew a \`paths:\` filter" "control: a paths filter that would deadlock PRs is caught" \
  python3 "$TMP/sg/scripts/assert-site-tests-gate.py"

sg_sabotage 'd["jobs"].pop("unit")'
want_rc_msg 1 "no longer exists" "control: deleting the suite outright is caught" \
  python3 "$TMP/sg/scripts/assert-site-tests-gate.py"

echo
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
