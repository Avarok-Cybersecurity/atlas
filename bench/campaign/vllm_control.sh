#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Render (and optionally run) the VERIFIED vLLM control leg for one
# (model, SKU) pair of the Hopper campaign.
#
# WHY THIS EXISTS. The control leg is only a control if its flags are the
# recipe's flags. Typing `vllm serve ...` by hand on a rented box is how a
# campaign acquires an unverifiable number: nobody can tell afterwards whether
# `--moe-backend flashinfer_cutlass` was in the recipe or in somebody's
# muscle memory. So this script owns exactly one job -- take a (model, SKU),
# look up the captured render in bench/campaign/vllm_recipes.json, and emit it
# unchanged. It composes nothing. If the pair has no rendered profile it says
# so and exits 3; it does not reconstruct one.
#
# THREE REFUSALS, EACH LOAD-BEARING
#
#  1. No rendered profile -> exit 3. `glm-4.5-air-fp8` is the live case: both
#     recipe twins 404'd on 2026-09-05 and the FP8 model card's command serves
#     the BF16 checkpoint. A reconstructed command is not a control.
#  2. A real run without VLLM_IMAGE_DIGEST -> exit 5. The artifact schema makes
#     the digest the engine's identity for vLLM (there is no `git_sha` for a
#     pip-installed Python server). A run whose image cannot be named later is
#     a number without provenance, so it is refused before it starts, not
#     annotated afterwards.
#  3. A real run of a MULTI-NODE profile -> exit 6. Kimi K3 (2-node TP8+PP2 on
#     B200, TP16 elsewhere), GLM-5.3 on H100 and MiniMax-M3 on H100 all render
#     as head + `--headless` worker across separate chassis. One `docker run`
#     on one host cannot be that; the head/worker pair is printed with its
#     `$HEAD_IP` / `--node-rank` placeholders for a cluster operator to place.
#
# SPECULATION IS BOTH-OR-NEITHER (bench/hopper_ab/workloads.json). `--spec` is
# therefore REQUIRED, never defaulted: the GB10 campaign's first vLLM reference
# ran spec OFF against an Atlas leg that ran it ON and every comparison drawn
# from it had to be retracted. `--spec on` adds exactly the recipe's
# `--speculative-config` tokens and `--spec off` removes exactly those; the
# selftest asserts the two renders differ by nothing else.
#
# WHAT A REAL RUN HANDS BACK. `docker run -d` prints the ID of the container it
# created, and that ID -- not the name -- is what a caller may later stop and
# remove. A name is a claim anybody can hold; an ID is proof this invocation
# created that container. So a successful run ends by printing
# `container_id: <id>` as its LAST line and, with --id-file, writing the bare
# ID to that path -- via a temp file and a rename, so a caller reading that
# path concurrently sees either no file or the whole ID, never a prefix of one.
# A `docker run` that fails (125 = the name is already in use) writes nothing:
# there is no container of ours to clean up, and the one holding the name is
# not ours to delete.
#
# Usage:
#   vllm_control.sh <model-key> <sku> --spec on|off [--dry-run] [--extra "..."]
#                   [--label KEY=VALUE] [--id-file PATH]
#   vllm_control.sh --list
#   vllm_control.sh --selftest
#
# Model keys: nemotron-3-super-fp8 nemotron-3-nano-fp8 qwen3.6-35b-a3b-fp8
#             qwen3-next-80b-fp8 deepseek-v4-flash qwen3.8-flash-next-fp8
#             glm-5.3 glm-5.3-flash glm-4.5-air-fp8 minimax-m3 kimi-k3
# SKUs:       h100 h200 b200 gb10
#
# Environment:
#   VLLM_IMAGE         override the recipe's image tag (default: the recipe's)
#   VLLM_IMAGE_DIGEST  sha256:... -- REQUIRED for a non-dry run; the reference
#                      becomes <repo>@<digest> so the tag cannot drift under it
#   HF_CACHE           host HF cache to mount (default ~/.cache/huggingface)
#   DOCKER             docker command (default "docker")
#   VLLM_CONTAINER     container name (default vllm-control-<model>-<sku>)
#   VLLM_RECIPES       path to the recipe data file (default: next to this script)
#
# Exit codes: 0 ok · 2 usage · 3 no rendered profile · 4 --spec on with no
#             speculative profile in the recipe · 5 run without a digest ·
#             6 run of a multi-node profile
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RECIPES="${VLLM_RECIPES:-$HERE/vllm_recipes.json}"
DOCKER_CMD="${DOCKER:-docker}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

MODEL_KEY=""
SKU=""
SPEC=""
DRY_RUN=0
EXTRA=""
ID_FILE=""
LABELS=()
LIST=0
SELFTEST=0

usage() { sed -n '2,61p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --spec) SPEC="${2:-}"; shift 2 ;;
    --extra) EXTRA="${2:-}"; shift 2 ;;
    --label) LABELS+=( --label "${2:-}" ); shift 2 ;;
    --id-file) ID_FILE="${2:-}"; shift 2 ;;
    --list) LIST=1; shift ;;
    --selftest) SELFTEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -z "$MODEL_KEY" ]; then MODEL_KEY="$1"
      elif [ -z "$SKU" ]; then SKU="$1"
      else echo "unexpected extra argument: $1" >&2; usage >&2; exit 2
      fi
      shift ;;
  esac
done

if [ ! -f "$RECIPES" ]; then
  echo "ERROR: recipe data file not found: $RECIPES" >&2
  exit 2
fi

# ── the renderer ─────────────────────────────────────────────────────────────
# Python owns the JSON and the flag arithmetic; bash owns process launch. The
# renderer writes NUL-separated argv files so a flag value containing spaces
# (every --reasoning-config / --attention-config JSON blob does) survives the
# hand-off without a quoting round trip.
render() {
  python3 "$HERE/vllm_render.py" "$@"
}

if [ "$SELFTEST" = "1" ]; then
  render --recipes "$RECIPES" --selftest
  exit $?
fi

if [ "$LIST" = "1" ]; then
  render --recipes "$RECIPES" --list
  exit $?
fi

if [ -z "$MODEL_KEY" ] || [ -z "$SKU" ]; then
  echo "ERROR: both <model-key> and <sku> are required." >&2
  usage >&2
  exit 2
fi

# Profile resolution runs BEFORE the --spec requirement so that "this pair was
# never rendered" is reported as itself (exit 3) rather than masked by a usage
# error about a flag that could not have helped.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

render --recipes "$RECIPES" --model "$MODEL_KEY" --sku "$SKU" --probe
probe_rc=$?
if [ "$probe_rc" -ne 0 ]; then
  exit "$probe_rc"
fi

case "$SPEC" in
  on|off) ;;
  "")
    echo "ERROR: --spec on|off is required." >&2
    echo "       Speculation is both-or-neither across the two engines" >&2
    echo "       (bench/hopper_ab/workloads.json); there is no safe default." >&2
    exit 2 ;;
  *) echo "ERROR: --spec must be 'on' or 'off', got '$SPEC'." >&2; exit 2 ;;
esac

CONTAINER="${VLLM_CONTAINER:-vllm-control-$(printf '%s' "$MODEL_KEY" | tr '.' '-')-$SKU}"

render --recipes "$RECIPES" --model "$MODEL_KEY" --sku "$SKU" --spec "$SPEC" \
       --stage "$STAGE" --container "$CONTAINER" --hf-cache "$HF_CACHE" \
       --docker "$DOCKER_CMD" --extra "$EXTRA" \
       ${LABELS[@]+"${LABELS[@]}"} \
       ${VLLM_IMAGE:+--image "$VLLM_IMAGE"} \
       ${VLLM_IMAGE_DIGEST:+--image-digest "$VLLM_IMAGE_DIGEST"}
rc=$?
if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi

if [ "$DRY_RUN" = "1" ]; then
  echo ""
  echo "dry-run: nothing launched."
  exit 0
fi

# ── real run: the two refusals ───────────────────────────────────────────────
if [ -z "${VLLM_IMAGE_DIGEST:-}" ]; then
  echo "" >&2
  echo "REFUSED: VLLM_IMAGE_DIGEST is unset." >&2
  echo "  A vLLM cell's engine identity IS its image digest -- the schema has no" >&2
  echo "  git_sha for a pip-installed Python server, and a floating tag can be" >&2
  echo "  re-pointed between the two legs of an A/B. Resolve it first:" >&2
  echo "    $DOCKER_CMD pull <image>" >&2
  echo "    $DOCKER_CMD inspect --format '{{index .RepoDigests 0}}' <image>" >&2
  echo "  then re-run with VLLM_IMAGE_DIGEST=sha256:..." >&2
  exit 5
fi

if [ -f "$STAGE/multinode" ]; then
  echo "" >&2
  echo "REFUSED: this profile is multi-node ($(cat "$STAGE/multinode") nodes)." >&2
  echo "  The head and worker commands above are printed with their \$HEAD_IP and" >&2
  echo "  --node-rank placeholders because a single 'docker run' on one host" >&2
  echo "  cannot be a multi-chassis deployment. Place them on the booked cluster" >&2
  echo "  by hand; --dry-run is the only supported mode here." >&2
  exit 6
fi

CMD=()
while IFS= read -r -d '' arg; do CMD+=( "$arg" ); done < "$STAGE/head.argv"

echo ""
echo "=== launching $CONTAINER ==="
CID="$("${CMD[@]}")"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "" >&2
  if [ "$rc" -eq 125 ]; then
    echo "docker run exited 125: the name '$CONTAINER' is already in use." >&2
    echo "  That container belongs to another run. NOTHING was created here, so" >&2
    echo "  nothing is recorded and nothing must be stopped or removed." >&2
  else
    echo "docker run exited $rc; no container was created by this invocation." >&2
  fi
  exit "$rc"
fi
CID="$(printf '%s' "$CID" | tail -1 | tr -d '[:space:]')"
if [ -z "$CID" ]; then
  echo "docker run succeeded but printed no container ID; refusing to guess one." >&2
  exit 5
fi
if [ -n "$ID_FILE" ]; then
  # Temp file then rename: the caller's teardown may read this path at any
  # moment, including while a signal is landing here, and a rename is the only
  # write it can observe as all-or-nothing. A half-written ID is worse than no
  # ID -- `docker stop <truncated>` is a stop of nothing while the container it
  # was meant to name keeps the GPU.
  printf '%s\n' "$CID" > "$ID_FILE.tmp.$$" && mv -f "$ID_FILE.tmp.$$" "$ID_FILE"
fi
echo "container_id: $CID"
