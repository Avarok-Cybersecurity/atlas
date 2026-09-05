#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# The whole bench/campaign/ suite. No GPU, no engine, no network: every case is
# a selftest, a --dry-run, or a refusal.
#
# What each block is actually defending:
#
#   (a) the three selftests -- vllm_control, atlas_render, validate_artifact --
#       each of which owns its own negatives. Run first, because everything
#       below assumes the data files are internally consistent.
#   (b) atlas --dry-run of a REAL cell: the ladder is invoked with the frozen
#       lat shape (isl 1024 / osl 256) from bench/hopper_ab/workloads.json, the
#       boot gate carries the 1800 s PRD cap, and the serve line carries the
#       recipe's flags. A shape that drifts from workloads.json silently is the
#       failure this catches -- two legs measured at different isl are not an
#       A/B, they are two numbers.
#   (c) vllm --dry-run at the agent shape (isl 4096 / osl 512), same checks. The
#       two workloads are tested on different engines on purpose: a bug that
#       hardcodes one shape survives testing both shapes on one engine.
#   (d) NO NCCL_* variable for an Atlas cell on H100. The gb10-roce block pins a
#       NIC that does not exist there, disables NVLink SHARP on a machine that
#       HAS NVLink, and forces Ring/Simple onto an intra-node transport. A stray
#       variable creeping back in is the regression, and it would be invisible
#       in the numbers. (NCCL_PROFILE is the launcher's own selector, not an
#       NCCL variable -- it is what SELECTS the empty profile.)
#   (e) the Kimi K3 refusal: a 2-node TP8+PP2 profile prints head and --headless
#       worker with their $HEAD_IP placeholders and REFUSES to run (exit 6). One
#       `docker run` on one host cannot be a two-chassis deployment.
#   (f) the missing-profile exit 3: glm-4.5-air-fp8 has no rendered recipe (both
#       twins 404'd), and a reconstructed command is not a control.
#   (g) source grep: no file here may contain an executable process-pattern kill
#       (`pkill -f`). Such a pattern matches the killing shell's own command
#       line; teardown here is by container name and by the launcher's pid
#       files. Comments explaining the ban are exempt, as in
#       scripts/start_node_ep_test.sh case (f).
#   (h) end-to-end assembly from stub stage outputs -> an artifact that
#       VALIDATES. The assembler is the piece with no natural test on a GPU-less
#       host, so it gets fed fixtures.
#   (i) lints: shellcheck on every .sh, py_compile on every .py, typos clean.
#
# Usage: bash bench/campaign/campaign_test.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUN="$HERE/run_cell.sh"
VC="$HERE/vllm_control.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

asserts=0
fail() { echo "ASSERT FAILED [$1]: $2" >&2; exit 1; }
ok() { asserts=$((asserts + 1)); echo "  ok [$1] $2"; }
have() { grep -Fq -- "$2" <<<"$1"; }

# ── (a) the three selftests ──────────────────────────────────────────────────
out="$(bash "$VC" --selftest 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail a "vllm_control --selftest exited $rc:
$out"
have "$out" "checks passed" || fail a "no pass line from vllm_control --selftest: $out"
ok a "vllm_control --selftest: $(tail -1 <<<"$out")"

out="$(python3 "$HERE/atlas_render.py" --recipes "$HERE/atlas_recipes.json" --selftest 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail a "atlas_render --selftest exited $rc:
$out"
ok a "atlas_render --selftest: $(tail -1 <<<"$out")"

out="$(python3 "$HERE/validate_artifact.py" --selftest 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail a "validate_artifact --selftest exited $rc:
$out"
ok a "validate_artifact --selftest: $(tail -1 <<<"$out")"

# ── (b) atlas dry run, lat shape ─────────────────────────────────────────────
atlas_out="$(bash "$RUN" --engine atlas --model nemotron-3-super-fp8 --sku h200 \
              --workload lat --concurrency 16 --spec off --think on \
              --out "$tmp/atlas-cell" --dry-run 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail b "atlas dry run exited $rc:
$atlas_out"
[ -e "$tmp/atlas-cell" ] && fail b "--dry-run must not create the output directory"
ok b "atlas --dry-run exits 0 and creates nothing"

ladder_line="$(grep -F 'harness_w55_conc_ladder.py' <<<"$atlas_out" | grep -F -- '--concs')"
[ -n "$ladder_line" ] || fail b "no ladder invocation in the atlas dry run:
$atlas_out"
have "$ladder_line" "--isl 1024 --osl 256" || fail b "lat shape must be isl 1024 / osl 256: $ladder_line"
have "$ladder_line" "--concs 16" || fail b "ladder must be driven at C=16: $ladder_line"
have "$ladder_line" "--reps 3" || fail b "ladder must run 3 reps: $ladder_line"
have "$ladder_line" "--warmup 1" || fail b "ladder must discard 1 warmup batch: $ladder_line"
ok b "ladder invoked at the frozen lat shape (isl 1024 / osl 256, C=16, reps 3, warmup 1)"

have "$atlas_out" "--timeout-s 1800" || fail b "boot gate must carry the 1800 s cap:
$atlas_out"
have "$atlas_out" "BOOT_TIMEOUT_S=1800" || fail b "launcher must carry the 1800 s cap:
$atlas_out"
ok b "boot gate and launcher both carry the 1800 s PRD cap"

have "$atlas_out" "--gpu-memory-utilization 0.92" || fail b "Super H200 override missing:
$atlas_out"
have "$atlas_out" "--tool-call-parser bare_json" || fail b "Super tool parser must be bare_json:
$atlas_out"
have "$atlas_out" "--kv-cache-dtype fp8" || fail b "campaign common KV dtype missing: $atlas_out"
ok b "serve line carries the recipe flags (util 0.92, bare_json, fp8 KV)"

have "$atlas_out" "WARNING: --think on sets a SERVE-side flag only." \
  || fail b "think-on must warn that the client pins enable_thinking=false:
$atlas_out"
ok b "--think on warns that the ladder client suppresses it"

# ── (c) vllm dry run, agent shape ────────────────────────────────────────────
vllm_out="$(bash "$RUN" --engine vllm --model qwen3.6-35b-a3b-fp8 --sku h100 \
             --workload agent --concurrency 1 --spec off --think off \
             --out "$tmp/vllm-cell" --dry-run 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail c "vllm dry run exited $rc:
$vllm_out"
[ -e "$tmp/vllm-cell" ] && fail c "--dry-run must not create the output directory"

ladder_line="$(grep -F 'harness_w55_conc_ladder.py' <<<"$vllm_out" | grep -F -- '--concs')"
[ -n "$ladder_line" ] || fail c "no ladder invocation in the vllm dry run:
$vllm_out"
have "$ladder_line" "--isl 4096 --osl 512" || fail c "agent shape must be isl 4096 / osl 512: $ladder_line"
have "$ladder_line" "--concs 1" || fail c "ladder must be driven at C=1: $ladder_line"
ok c "ladder invoked at the frozen agent shape (isl 4096 / osl 512, C=1)"

have "$vllm_out" "--timeout-s 1800" || fail c "boot gate must carry the 1800 s cap:
$vllm_out"
ok c "vllm boot gate carries the 1800 s PRD cap"

have "$vllm_out" "docker run" || fail c "no docker run line in the vllm dry run:
$vllm_out"
have "$vllm_out" "--gpus all --ipc=host --network host" || fail c "docker preamble wrong:
$vllm_out"
have "$vllm_out" "a non-dry run will be REFUSED, exit 5" \
  || fail c "a tag-only image must warn that a real run is refused:
$vllm_out"
ok c "vllm renders the recipe under docker run and flags the missing digest"

# ── (d) no NCCL_* variable for Atlas on H100 ─────────────────────────────────
# NCCL_PROFILE is the launcher's selector for WHICH profile to emit; the
# pessimizing variables themselves are what must never appear.
h100_out="$(bash "$RUN" --engine atlas --model nemotron-3-super-fp8 --sku h100 \
             --workload lat --concurrency 16 --spec off --think off \
             --out "$tmp/h100-cell" --dry-run 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail d "atlas h100 dry run exited $rc:
$h100_out"
stray="$(grep -oE 'NCCL_(SOCKET_IFNAME|IB_[A-Z_]+|IB_DISABLE|NVLS_ENABLE|NET_GDR[A-Z_]*|DMABUF_ENABLE|CUMEM_HOST_ENABLE|PROTO|ALGO|MIN_NCHANNELS|MAX_NCHANNELS|DEBUG|DEBUG_SUBSYS)=' <<<"$h100_out" | sort -u)"
[ -z "$stray" ] || fail d "an Atlas H100 cell must ship NO NCCL configuration, found:
$stray"
have "$h100_out" "NCCL_PROFILE=default" || fail d "the launcher must be told to use the empty profile:
$h100_out"
ok d "Atlas on H100 emits no NCCL_* setting (only NCCL_PROFILE=default selects the empty profile)"

have "$h100_out" "--rank 0 --world-size 2 --ep-size 2 --tp-size 1" \
  || fail d "Super H100 must be EP=2 across two ranks:
$h100_out"
have "$h100_out" "--rank 1 --world-size 2 --ep-size 2 --tp-size 1" \
  || fail d "Super H100 rank 1 missing:
$h100_out"
have "$h100_out" "--disable-thinking" || fail d "think off must add --disable-thinking:
$h100_out"
ok d "Super on H100 renders both EP=2 ranks and the think-off flag"

# ── (e) the Kimi K3 multi-node refusal ───────────────────────────────────────
k3_dry="$(bash "$VC" kimi-k3 b200 --spec off --dry-run 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail e "kimi-k3 dry run exited $rc:
$k3_dry"
have "$k3_dry" "tp=8 pp=2" || fail e "kimi-k3 b200 must render 2-node TP8+PP2: $k3_dry"
have "$k3_dry" "--node-rank 0" || fail e "no head node-rank in the K3 render"
have "$k3_dry" "--node-rank 1" || fail e "no worker node-rank in the K3 render"
have "$k3_dry" "--headless" || fail e "the K3 worker must be --headless"
have "$k3_dry" '--master-addr ' "$k3_dry" || true
grep -Fq -- "'\$HEAD_IP'" <<<"$k3_dry" || fail e "the \$HEAD_IP placeholder must survive verbatim"
ok e "kimi-k3 b200 prints head + --headless worker with \$HEAD_IP / --node-rank placeholders"

VLLM_IMAGE_DIGEST="sha256:$(printf 'a%.0s' $(seq 64))" bash "$VC" kimi-k3 b200 --spec off \
  >/dev/null 2>"$tmp/k3.err"; rc=$?
[ $rc -eq 6 ] || fail e "a real K3 run must exit 6, got $rc:
$(cat "$tmp/k3.err")"
grep -q "REFUSED: this profile is multi-node" "$tmp/k3.err" \
  || fail e "the K3 refusal must say why: $(cat "$tmp/k3.err")"
ok e "a real Kimi K3 run is refused with exit 6"

bash "$VC" nemotron-3-nano-fp8 h100 --spec off >/dev/null 2>"$tmp/nodigest.err"; rc=$?
[ $rc -eq 5 ] || fail e "a real run with no digest must exit 5, got $rc"
ok e "a real run with no VLLM_IMAGE_DIGEST is refused with exit 5"

bash "$VC" nemotron-3-nano-fp8 h100 --spec on --dry-run >/dev/null 2>"$tmp/nospec.err"; rc=$?
[ $rc -eq 4 ] || fail e "--spec on with no speculative profile must exit 4, got $rc"
ok e "--spec on against a recipe with no speculative profile is refused with exit 4"

# ── (f) missing profile -> exit 3 ────────────────────────────────────────────
out="$(bash "$VC" glm-4.5-air-fp8 h100 --spec off --dry-run 2>&1)"; rc=$?
[ $rc -eq 3 ] || fail f "vllm_control on a pair with no profile must exit 3, got $rc: $out"
have "$out" "no rendered profile for glm-4.5-air-fp8 on h100" \
  || fail f "exit 3 must name the pair: $out"
ok f "vllm_control: no rendered profile -> exit 3, naming the pair"

out="$(bash "$RUN" --engine vllm --model nemotron-3-nano-fp8 --sku gb10 --workload lat \
        --concurrency 1 --spec off --think off --out "$tmp/nope" --dry-run 2>&1)"; rc=$?
[ $rc -eq 3 ] || fail f "run_cell on a pair with no vLLM profile must exit 3, got $rc: $out"
ok f "run_cell: no vLLM profile for gb10 -> exit 3"

out="$(bash "$RUN" --engine atlas --model kimi-k3 --sku h200 --workload lat \
        --concurrency 1 --spec off --think off --out "$tmp/nope" --dry-run 2>&1)"; rc=$?
[ $rc -eq 3 ] || fail f "run_cell on a model with no Atlas recipe must exit 3, got $rc: $out"
ok f "run_cell: no Atlas recipe for kimi-k3 -> exit 3"

out="$(bash "$RUN" --engine atlas --model nemotron-3-nano-fp8 --sku h100 --workload lat \
        --concurrency 1 --spec off --think off --out "$tmp/nope" 2>&1)"; rc=$?
[ $rc -eq 2 ] || fail f "run_cell without --yes or --dry-run must exit 2, got $rc: $out"
have "$out" "REFUSED: this would start an engine on this box." \
  || fail f "the refusal must say what it is refusing: $out"
[ -e "$tmp/nope" ] && fail f "a refused run must not create its output directory"
ok f "run_cell without --yes or --dry-run refuses to start anything (exit 2)"

# ── (g) no executable process-pattern kill anywhere under bench/campaign ─────
# The pattern is assembled from parts so this file does not itself contain the
# literal string it bans -- a check that fails on its own source invites the
# "fix" of weakening the check. Comment lines are excluded the same way
# scripts/start_node_ep_test.sh case (f) does it: the whole point of the ban is
# that it be explained in prose next to the code that obeys it.
PK="pkill"
BANNED="$PK -f"
hits="$(grep -rn --exclude-dir=__pycache__ --binary-files=without-match -- "$BANNED" "$HERE" \
        | grep -v '^[^:]*:[0-9]*:[[:space:]]*#' || true)"
[ -z "$hits" ] || fail g "an executable '$BANNED' exists under bench/campaign:
$hits"
ok g "no executable '$BANNED' under bench/campaign (teardown is by container name and pid file)"

# ── (h) assemble from stubs -> a VALID artifact ──────────────────────────────
art="$tmp/assembled.json"
python3 "$HERE/cell_assemble.py" --engine atlas --model-key nemotron-3-super-fp8 \
  --sku h200 --workload lat --concurrency 16 --spec off --think on --out "$art" \
  --workloads "$ROOT/bench/hopper_ab/workloads.json" \
  --atlas-recipes "$HERE/atlas_recipes.json" --vllm-recipes "$HERE/vllm_recipes.json" \
  --client "$ROOT/bench/ladder38/harness_w55_conc_ladder.py" \
  --nvidia-smi-q "$HERE/fixtures/stub_nvidia_smi_q.txt" \
  --boot-json "$HERE/fixtures/stub_boot.json" \
  --coherency-json "$HERE/fixtures/stub_coherency.json" \
  --ladder-json "$HERE/fixtures/stub_ladder_c16.json" \
  --git-sha "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo 0000000)" \
  >/dev/null 2>&1 || fail h "cell_assemble failed on the stub stage outputs"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail h "the assembled artifact does not validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
ok h "cell_assemble turns the stub stage outputs into an artifact that validates"

read -r verdict stage method thinkv <<<"$(python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(d["verdict"], d["failing_stage"], d["metrics"]["percentile_method"], d["think"])' "$art")"
[ "$verdict" = "PARTIAL" ] || fail h "an unpaired all-green cell must be PARTIAL, got $verdict"
[ "$stage" = "pair" ] || fail h "an unpaired cell must name the 'pair' stage, got $stage"
[ "$method" = "mean_of_rep_percentiles" ] \
  || fail h "a ladder-derived cell must not claim pooled percentiles, got $method"
ok h "unpaired all-green cell is PARTIAL/pair with mean_of_rep_percentiles"

python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
assert "enable_thinking=false" in d["notes"], d["notes"]
' "$art" || fail h "a think-on cell must record the client caveat in notes"
[ "$thinkv" = "on" ] || fail h "think must round-trip into the artifact, got $thinkv"
ok h "the think-on caveat reaches the artifact notes"

# A failing stage still produces an artifact, with the verdict that says so.
python3 "$HERE/cell_assemble.py" --engine atlas --model-key nemotron-3-super-fp8 \
  --sku h200 --workload lat --concurrency 16 --spec off --think off \
  --out "$tmp/nogo.json" --workloads "$ROOT/bench/hopper_ab/workloads.json" \
  --atlas-recipes "$HERE/atlas_recipes.json" --vllm-recipes "$HERE/vllm_recipes.json" \
  --client "$ROOT/bench/ladder38/harness_w55_conc_ladder.py" \
  --failing-stage boot >/dev/null 2>&1 || fail h "cell_assemble failed for a NO-GO cell"
python3 "$HERE/validate_artifact.py" "$tmp/nogo.json" >/dev/null \
  || fail h "a NO-GO artifact must still validate:
$(python3 "$HERE/validate_artifact.py" "$tmp/nogo.json")"
v="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$tmp/nogo.json")"
[ "$v" = "NO-GO" ] || fail h "a failed boot gate must be NO-GO, got $v"
ok h "a boot-gate failure still writes a VALID artifact, verdict NO-GO"

# ── (i) lints ────────────────────────────────────────────────────────────────
if command -v shellcheck >/dev/null 2>&1; then
  for f in "$HERE"/*.sh; do
    shellcheck "$f" || fail i "shellcheck failed on $f"
  done
  ok i "shellcheck clean on every .sh under bench/campaign"
else
  echo "  -- [i] shellcheck not installed, skipped"
fi

for f in "$HERE"/*.py; do
  python3 -m py_compile "$f" || fail i "py_compile failed on $f"
done
rm -rf "$HERE/__pycache__"
ok i "python3 -m py_compile clean on every .py under bench/campaign"

for f in "$HERE"/*.json "$HERE"/fixtures/*.json; do
  python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$f" \
    || fail i "$f is not valid JSON"
done
ok i "every .json under bench/campaign parses"

if command -v typos >/dev/null 2>&1; then
  typos "$HERE" || fail i "typos found under bench/campaign"
  ok i "typos clean under bench/campaign"
else
  echo "  -- [i] typos not installed, skipped"
fi

echo ""
echo "ALL $asserts assertions passed."
