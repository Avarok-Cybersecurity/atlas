#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Tests for docker/hopper/Dockerfile, docker/b200/Dockerfile and
# .github/workflows/datacenter-binaries.yml. No Docker daemon, no GPU, no
# network: every case is a structural assertion over files that are checked in.
#
# WHY A TEST AND NOT A REVIEW. None of these three artefacts has ever been
# executed — the images take about an hour to build and need an x86_64 nvcc,
# and the workflow has never been dispatched. What CAN be pinned cheaply is the
# set of properties whose violation would be silent, discovered only on a
# rented GPU:
#
#   (a) the default ATLAS_TARGET_HW per file. Copy one Dockerfile from the
#       other and forget this one line and you get two identical images under
#       two names — a "b200" image full of sm_90a PTX that the arch preflight
#       refuses on a B200, an hour of build time after the mistake.
#   (b) no NCCL_* environment baked in. scripts/start-ep2.sh's block is tuned
#       for two GB10 chassis over RoCE: it names a NIC these boxes do not
#       have, disables NVLink SHARP, and forces the slowest protocol/algorithm
#       pair onto an intra-node transport. On one NVLink node the right NCCL
#       configuration is no NCCL configuration, and a stray ENV would apply it
#       to every rank silently. Same regression scripts/start_node_ep_test.sh
#       case (a) guards for the launcher.
#   (c) the NCCL >= 2.28 gate in the runtime stage. Below 2.28 there is no
#       ncclMemAlloc/ncclMemFree symmetric memory, and the failure surfaces as
#       a runtime error inside a fused allreduce, not as a bad image.
#   (d) --features cuda,nccl. Dropping `nccl` still builds and still serves —
#       single GPU. Multi-rank EP/TP is what silently disappears.
#   (e) the workflow's matrix really covers both hardware sets, and its
#       dispatch input offers both. A `both` that resolves to one leg would
#       produce a green run and one artifact.
#
# Usage: bash docker/datacenter_dockerfiles_test.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOPPER="$ROOT/docker/hopper/Dockerfile"
B200="$ROOT/docker/b200/Dockerfile"
WORKFLOW="$ROOT/.github/workflows/datacenter-binaries.yml"

asserts=0
fail() { echo "ASSERT FAILED [$1]: $2" >&2; exit 1; }
ok() { asserts=$((asserts + 1)); echo "  ok [$1] $2"; }

for f in "$HOPPER" "$B200" "$WORKFLOW"; do
  [ -f "$f" ] || fail setup "missing file: $f"
done

# Dockerfile instructions only — comments in these files legitimately discuss
# NCCL_SOCKET_IFNAME, ATLAS_TARGET_HW and the rest, and a naive grep over the
# raw text would pass on prose and fail on nothing.
instructions() { sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$1"; }

# ── (a) each file defaults to its own hardware set ────────────────────────────
for pair in "hopper:$HOPPER" "b200:$B200"; do
  hw="${pair%%:*}"; f="${pair#*:}"
  got="$(instructions "$f" | sed -n 's/^[[:space:]]*ARG[[:space:]]\{1,\}ATLAS_TARGET_HW=\(.*\)$/\1/p')"
  [ -n "$got" ] || fail a "$f declares no \`ARG ATLAS_TARGET_HW=<default>\`"
  [ "$(printf '%s\n' "$got" | wc -l | tr -d ' ')" = 1 ] \
    || fail a "$f declares ARG ATLAS_TARGET_HW more than once: $got"
  [ "$got" = "$hw" ] || fail a "$f defaults ATLAS_TARGET_HW to '$got', expected '$hw'"
  ok a "$(basename "$(dirname "$f")")/Dockerfile defaults ATLAS_TARGET_HW=$hw"

  # The ARG has to reach the build environment, or it is decoration.
  instructions "$f" | grep -qE '^[[:space:]]*ENV[[:space:]]+ATLAS_TARGET_HW=\$\{?ATLAS_TARGET_HW\}?[[:space:]]*$' \
    || fail a "$f never exports ATLAS_TARGET_HW as an ENV from its ARG"
  ok a "$(basename "$(dirname "$f")")/Dockerfile exports ATLAS_TARGET_HW to the build"

  # '*' = every model target under kernels/<hw>/.
  instructions "$f" | grep -qE "^[[:space:]]*ARG[[:space:]]+ATLAS_TARGET_MODEL=[\"']?\*[\"']?[[:space:]]*$" \
    || fail a "$f does not default ATLAS_TARGET_MODEL to '*'"
  ok a "$(basename "$(dirname "$f")")/Dockerfile defaults ATLAS_TARGET_MODEL='*'"
done

# ── (b) no NCCL_* environment is baked into either image ──────────────────────
for f in "$HOPPER" "$B200"; do
  stray="$(instructions "$f" | grep -nE '^[[:space:]]*ENV[[:space:]].*NCCL_' || true)"
  [ -z "$stray" ] || fail b "$f bakes in NCCL environment:
$stray"
  ok b "$(basename "$(dirname "$f")")/Dockerfile bakes in no NCCL_* env"
done

# ── (c) the runtime stage refuses NCCL < 2.28 ────────────────────────────────
for f in "$HOPPER" "$B200"; do
  grep -q 'dpkg --compare-versions "$NCCL_VER" ge "2.28"' "$f" \
    || fail c "$f has no \`dpkg --compare-versions … ge 2.28\` gate"
  grep -q 'exit 1' "$f" || fail c "$f's NCCL gate does not fail the build"
  ok c "$(basename "$(dirname "$f")")/Dockerfile gates NCCL >= 2.28 and exits non-zero below it"
done

# ── (d) the build keeps the multi-GPU feature set, and ships `spark` ──────────
for f in "$HOPPER" "$B200"; do
  grep -q -- '--no-default-features --features cuda,nccl' "$f" \
    || fail d "$f does not build with --no-default-features --features cuda,nccl"
  ok d "$(basename "$(dirname "$f")")/Dockerfile builds --features cuda,nccl"

  instructions "$f" | grep -qE '^[[:space:]]*ENTRYPOINT[[:space:]]+\["spark"\][[:space:]]*$' \
    || fail d "$f does not ENTRYPOINT [\"spark\"]"
  ok d "$(basename "$(dirname "$f")")/Dockerfile entrypoints spark"

  # Base images pinned by digest, both stages. A floating tag would change the
  # CUDA toolkit under a build that has never been re-verified.
  n="$(instructions "$f" | grep -cE '^[[:space:]]*FROM[[:space:]]+nvidia/cuda:[^[:space:]]+@sha256:[0-9a-f]{64}')"
  [ "$n" = 2 ] || fail d "$f has $n digest-pinned FROM lines, expected 2"
  ok d "$(basename "$(dirname "$f")")/Dockerfile pins both base images by digest"
done

# ── (e) the workflow parses, and its matrix covers both hardware sets ─────────
# PyYAML where available (a real parse); otherwise a strict text check, so this
# test never silently degrades to nothing and never fails for a missing dep.
#
# The python is written to a file and then run, rather than heredoc'd straight
# into `$( … )`. macOS still ships bash 3.2, whose command-substitution parser
# scans heredoc bodies for quotes — one apostrophe inside a python comment is
# enough to make the whole script die with "unexpected EOF while looking for
# matching `''". Costing a temp file is cheaper than that class of bug.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cat > "$tmp/check_workflow.py" <<'PYEOF'
import json, os, re, sys, yaml

path = os.environ["WORKFLOW"]
doc = yaml.safe_load(open(path))

# `on:` is the YAML 1.1 boolean True unless quoted — safe_load gives us that
# key, so accept either spelling rather than depending on the file quoting.
on = doc.get("on", doc.get(True))
if on is None:
    sys.exit("workflow declares no triggers")

opts = on["workflow_dispatch"]["inputs"]["hw"]["options"]
if sorted(opts) != ["b200", "both", "hopper"]:
    sys.exit(f"hw input options are {opts}, expected hopper/b200/both")

paths = on["pull_request"]["paths"]
for want in ("kernels/hopper/**", "kernels/b200/**"):
    if want not in paths:
        sys.exit(f"pull_request paths missing {want}: {paths}")

jobs = doc["jobs"]
for want in ("plan", "build"):
    if want not in jobs:
        sys.exit(f"no {want} job")

# The matrix is produced by the plan job shell, so read the JSON it would emit
# for `both` and check the legs, rather than trusting the surrounding prose.
# This is the assertion that would catch a `both` that builds one arch.
script = "".join(s.get("run", "") for s in jobs["plan"]["steps"])
m = re.search(r"both\)\s+sel='(\{.*?\})'", script)
if not m:
    sys.exit("plan job has no both) branch emitting a matrix JSON")
legs = json.loads(m.group(1))["include"]
hws = sorted(leg["hw"] for leg in legs)
if hws != ["b200", "hopper"]:
    sys.exit(f"both resolves to {hws}, expected both hopper and b200")
arches = {leg["hw"]: leg["arch"] for leg in legs}
if arches != {"hopper": "sm_90a", "b200": "sm_100a"}:
    sys.exit(f"matrix arches are {arches}, expected hopper=sm_90a b200=sm_100a")

# The workflow MENTIONS ATLAS_SKIP_BUILD in a comment on purpose (not set, and
# that is the point), so strip comments before looking for an assignment.
# Setting it would emit the stub registry that build.rs writes without nvcc,
# and ship a binary with no kernels — green, downloadable, and useless.
body = "\n".join(re.sub(r"#.*$", "", line) for line in open(path).read().splitlines())
if re.search(r"ATLAS_SKIP_BUILD\s*[:=]", body):
    sys.exit("the workflow sets ATLAS_SKIP_BUILD — that would ship a stub PTX registry")

print("workflow ok: both legs, hopper=sm_90a b200=sm_100a")
PYEOF

if python3 -c 'import yaml' 2>/dev/null; then
  mode=yaml
  out="$(WORKFLOW="$WORKFLOW" python3 "$tmp/check_workflow.py" 2>&1)"     || fail e "$out"
  ok e "$out"
else
  mode=text
  grep -qE '^[[:space:]]*options: \[hopper, b200, both\]$' "$WORKFLOW" \
    || fail e "hw input does not offer hopper/b200/both"
  for want in "kernels/hopper/\*\*" "kernels/b200/\*\*"; do
    grep -q "$want" "$WORKFLOW" || fail e "pull_request paths missing $want"
  done
  # Branch-aware on purpose: a plain file-wide grep for each leg passes on a
  # `both)` that emits only one of them, because the single-hw branches carry
  # the other string. Read the `both)` line itself.
  both_line="$(grep -E '^[[:space:]]*both\)[[:space:]]+sel=' "$WORKFLOW" || true)"
  [ -n "$both_line" ] || fail e "the plan job has no both) branch"
  case "$both_line" in
    *'"hw":"hopper","arch":"sm_90a"'*) ;;
    *) fail e "both) omits the hopper/sm_90a leg: $both_line" ;;
  esac
  case "$both_line" in
    *'"hw":"b200","arch":"sm_100a"'*) ;;
    *) fail e "both) omits the b200/sm_100a leg: $both_line" ;;
  esac
  # Comments mention it on purpose; only an assignment is a bug.
  if sed 's/#.*$//' "$WORKFLOW" | grep -qE 'ATLAS_SKIP_BUILD[[:space:]]*[:=]'; then
    fail e "the workflow sets ATLAS_SKIP_BUILD — that would ship a stub PTX registry"
  fi
  ok e "workflow covers both hardware sets (text mode — PyYAML unavailable)"
fi

echo
echo "PASS — $asserts assertions (workflow checked in $mode mode)"
