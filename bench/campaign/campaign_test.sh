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
#   (k) CERTIFIED eligibility. The verdict has to come from gate EVIDENCE and
#       from a pair that is actually this cell's other leg. Any parseable
#       --paired-artifact used to be enough -- `{}` included -- and the three
#       measurement gates the ladder reports while exiting 0 (the 80% vacuity
#       floor, request errors, the 10% spread) only reached `notes`. Each red
#       below is one of those: an empty pair, a pair on the wrong SKU, a pair
#       three days old, a failed boot gate, a failed known-answer probe, and
#       ladder JSONs that exit 0 while failing each measurement gate.
#   (j) teardown ownership, against a PATH-shimmed `docker` that records every
#       call. A `docker run` that exits 125 (the name is already in use) means
#       somebody else owns that container, and the cell must stop and remove
#       NOTHING -- the old teardown ran `docker stop <name>` / `docker rm
#       <name>` unconditionally and deleted the other run's live server. The
#       cell tears down by the container ID its own `docker run -d` returned,
#       and the name carries this invocation's pid so the collision is
#       unlikely in the first place.
#   (l) the same ownership, on the way out of a SIGNAL. The teardown marked
#       "always" was reached by normal control flow only: a cell killed during
#       its boot poll exited -15 with the container it had just created still
#       running and no artifact at all, while the otherwise identical boot
#       FAILURE stopped and removed that same container and wrote its NO-GO.
#       Same owned resources, opposite outcome -- and on a rented box the
#       difference is a vLLM server holding the GPU until somebody finds it.
#
#   (m) ownership during the CREATE itself. `docker run -d` makes the
#       container and then prints its ID, and a cell killed between the two
#       held no ID to tear down by: the stub's running set kept the container
#       and the Docker log showed only `run`. Ownership has to be recoverable
#       from what was chosen and recorded before the create.
#   (n) ownership of an atlas launch that was REFUSED. The wrapper set its
#       "this launch is mine to stop" flag the moment it backgrounded the
#       launcher, so a launcher that refused an occupied port before creating
#       anything still got a --stop -- against a run directory whose records
#       belonged to an earlier invocation, whose rank it then killed.
#   (o) the same create window as (m), on the ATLAS side. The launcher records
#       rank<N>.container only after `docker run -d` returns, so a cell killed
#       inside the create owned a rank by no record at all and stopped
#       nothing. The intent written before the create is what makes it
#       recoverable, and the query that reconciles it carries the exact name
#       AND this launch's label -- a container of another run wearing that
#       name is untouched.
#   (r) engine identity under a MUTABLE tag. capture_engine_identity ran after
#       teardown and inspected the image TAG, so a rebuild that re-pointed the
#       tag during the run made the artifact name a build that never served a
#       request. The identity is the launched container's resolved image ID,
#       recorded at create time and read before teardown removes it.
#   (q) a teardown lookup that FAILED, mirrored from the launcher's case (l):
#       `docker ps --filter label=... || true` read an unreachable daemon as
#       "nothing was created here", and the cell finished its teardown stage
#       cleanly while its container kept the GPU.
#   (p) engine_version is the ENGINE's identity. The checkout SHA was passed
#       as --git-sha for either engine and a digest was captured only for
#       vLLM, so an Atlas container cell claimed the campaign's revision with
#       neither a digest nor a binary hash -- and certified. Each engine now
#       records what actually ran, and the checkout SHA is `harness`.
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

# A nested refusal must remain nonzero after the dry-run finalizer.
python3 "$HERE/dryrun_failure_test.py" || fail dryrun "renderer failure regression"
ok dryrun "3 dry-run exit regressions passed (both engines, refusal before boot)"
python3 "$HERE/thinking_policy_test.py" || fail policy "thinking policy regression"
ok policy "campaign thinking eligibility refuses excluded modes before launch"

# ── (a0) every FP8-checkpoint Atlas entry carries FP8 KV calibration ────────
# Oracle: spark-runtime/src/weights.rs fp8_kv_scale_count — a checkpoint with
# no *.k_scale gets scale 1.0 and clips; the GB10 rehearsal (2026-09-05) saw
# degenerate output from exactly that. 256 is qwen3.6-35b-a3b's MODEL.toml
# value for its FP8 checkpoint.
python3 - "$HERE/atlas_recipes.json" <<'PY' || fail a0 "an FP8 Atlas entry lacks --fp8-kv-calibration-tokens"
import json, sys
d = json.load(open(sys.argv[1]))
bad = [e["model_key"] + "/" + e["sku"] for e in d["entries"]
       if e.get("quant") == "fp8" and "--fp8-kv-calibration-tokens" not in e["extra_args"]]
assert not bad, bad
PY
ok a0 "every FP8 Atlas entry carries --fp8-kv-calibration-tokens"

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

have "$atlas_out" "--warmup 1 --enable-thinking" \
  || fail b "think-on must pass --enable-thinking to the ladder client:
$atlas_out"
ok b "--think on reaches the ladder client as --enable-thinking"

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
  --harness-git-sha "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo 0000000)" \
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
assert "enable_thinking=False" in d["notes"] and "not evidence of thinking-enabled" in d["notes"], d["notes"]
' "$art" || fail h "a think-on cell whose ladder header says false must say so in notes"
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

# ── (k) CERTIFIED comes from the gates and from a real pair ─────────────────
# One assemble helper: everything is the all-green cell except what a case
# overrides, so each red is exactly one changed input.
# A stand-in for the engine binary the cell would have run: a CERTIFIED cell
# has to identify the build that produced its numbers, and for a local-binary
# Atlas cell that identity is the hash of the file that ran.
printf 'not really the spark binary\n' > "$tmp/fake-spark"
k_assemble() {  # k_assemble OUT [extra cell_assemble args...]
  local out="$1"; shift
  python3 "$HERE/cell_assemble.py" --engine atlas --model-key nemotron-3-super-fp8 \
    --sku h200 --workload lat --concurrency 16 --spec off --think off --out "$out" \
    --workloads "$ROOT/bench/hopper_ab/workloads.json" \
    --atlas-recipes "$HERE/atlas_recipes.json" --vllm-recipes "$HERE/vllm_recipes.json" \
    --client "$ROOT/bench/ladder38/harness_w55_conc_ladder.py" \
    --nvidia-smi-q "$HERE/fixtures/stub_nvidia_smi_q.txt" \
    --boot-json "${K_BOOT:-$HERE/fixtures/stub_boot.json}" \
    --coherency-json "${K_COH:-$HERE/fixtures/stub_coherency.json}" \
    --ladder-json "${K_LADDER:-$HERE/fixtures/stub_ladder_c16.json}" \
    --binary "$tmp/fake-spark" \
    "$@" >/dev/null 2>&1
}
k_verdict() {
  python3 -c 'import json,sys
d = json.load(open(sys.argv[1]))
print(d["verdict"], d["failing_stage"], d["paired_cell"]["within_24h"])' "$1"
}
GOOD_PAIR="$HERE/fixtures/stub_pair_vllm_cell.json"

k_assemble "$tmp/k-certified.json" --paired-artifact "$GOOD_PAIR" \
  || fail k "cell_assemble failed on the all-green inputs"
read -r v st w <<<"$(k_verdict "$tmp/k-certified.json")"
[ "$v" = "CERTIFIED" ] || fail k "an all-green paired cell must be CERTIFIED, got $v ($st)"
[ "$st" = "None" ] || fail k "a CERTIFIED cell names no stage, got $st"
[ "$w" = "True" ] || fail k "a CERTIFIED cell records within_24h=true, got $w"
python3 "$HERE/validate_artifact.py" "$tmp/k-certified.json" >/dev/null \
  || fail k "the CERTIFIED artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$tmp/k-certified.json")"
ok k "an all-green cell with its real pair is CERTIFIED and validates"

# Each red: (label, extra assemble args...) -> the verdict and stage it owes.
k_red() {  # k_red LABEL EXPECT_VERDICT EXPECT_STAGE -- ARGS...
  local label="$1" want_v="$2" want_st="$3"; shift 4
  local out="$tmp/k-$label.json"
  k_assemble "$out" "$@" || fail k "$label: cell_assemble failed"
  read -r v st _w <<<"$(k_verdict "$out")"
  [ "$v" = "$want_v" ] || fail k "$label: expected $want_v, got $v (stage $st)"
  [ "$st" = "$want_st" ] || fail k "$label: expected stage $want_st, got $st"
  python3 "$HERE/validate_artifact.py" "$out" >/dev/null \
    || fail k "$label: the artifact must still validate:
$(python3 "$HERE/validate_artifact.py" "$out")"
}

k_red no-pair PARTIAL pair --
k_red empty-pair PARTIAL pair -- --paired-artifact "$HERE/fixtures/stub_pair_empty.json"
k_red wrong-sku-pair PARTIAL pair -- --paired-artifact "$HERE/fixtures/stub_pair_wrong_sku.json"
k_red stale-pair PARTIAL pair -- --paired-artifact "$HERE/fixtures/stub_pair_stale.json"
ok k "no pair, an empty {} pair, a pair on another SKU and a three-day-old pair are all PARTIAL/pair"

w="$(k_verdict "$tmp/k-empty-pair.json" | awk '{print $3}')"
[ "$w" = "False" ] || fail k "an empty pair must record within_24h=false, got $w"
w="$(k_verdict "$tmp/k-stale-pair.json" | awk '{print $3}')"
[ "$w" = "False" ] || fail k "a stale pair must record within_24h=false, got $w"
have "$(cat "$tmp/k-stale-pair.json")" "outside the 24 h window" \
  || fail k "the stale pair must say why: $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["notes"])' "$tmp/k-stale-pair.json")"
ok k "a rejected pair records within_24h=false and names the reason"

K_BOOT="$HERE/fixtures/stub_boot_timeout.json" \
  k_red failed-boot NO-GO boot -- --paired-artifact "$GOOD_PAIR"
K_COH="$HERE/fixtures/stub_coherency_wrong_answer.json" \
  k_red failed-coherency NO-GO coherency -- --paired-artifact "$GOOD_PAIR"
k_red absent-gates NO-GO boot -- --paired-artifact "$GOOD_PAIR" \
  --boot-json /nonexistent --coherency-json /nonexistent --ladder-json /nonexistent
ok k "a boot timeout, a failed known-answer probe and absent gate JSONs are NO-GO, never CERTIFIED"

K_LADDER="$HERE/fixtures/stub_ladder_c16_vacuous.json" \
  k_red vacuous-ladder PARTIAL ladder -- --paired-artifact "$GOOD_PAIR"
K_LADDER="$HERE/fixtures/stub_ladder_c16_errors.json" \
  k_red errored-ladder PARTIAL ladder -- --paired-artifact "$GOOD_PAIR"
K_LADDER="$HERE/fixtures/stub_ladder_c16_spread.json" \
  k_red spread-ladder PARTIAL ladder -- --paired-artifact "$GOOD_PAIR"
ok k "a ladder that exits 0 while failing the vacuity, error or spread gate is PARTIAL/ladder"

# The validator must refuse the claim on its own, not only trust the assembler.
python3 - "$tmp/k-certified.json" "$tmp/k-forged.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["metrics"]["vacuous"] = None
d["coherency"]["known_answer_ok"] = None
d["paired_cell"] = {"cell_id": None, "artifact_path": None, "within_24h": None}
json.dump(d, open(sys.argv[2], "w"), indent=2)
PY
python3 "$HERE/validate_artifact.py" "$tmp/k-forged.json" >"$tmp/k-forged.err" 2>&1 \
  && fail k "a CERTIFIED artifact with null gates and no pair must be REJECTED"
for path in "\$.coherency.known_answer_ok" "\$.metrics.vacuous" "\$.paired_cell.within_24h"; do
  grep -Fq -- "$path" "$tmp/k-forged.err" \
    || fail k "the rejection must name $path: $(cat "$tmp/k-forged.err")"
done
ok k "validate_artifact refuses a CERTIFIED verdict its own gate values do not support"

# ── (k2) --spec off records method=none even for a spec-capable recipe ───────
python3 "$HERE/cell_assemble.py" --engine atlas --model-key qwen3.6-35b-a3b-fp8 \
  --sku h200 --workload lat --concurrency 16 --spec off --think off \
  --out "$tmp/specoff.json" --workloads "$ROOT/bench/hopper_ab/workloads.json" \
  --atlas-recipes "$HERE/atlas_recipes.json" --vllm-recipes "$HERE/vllm_recipes.json" \
  --client "$ROOT/bench/ladder38/harness_w55_conc_ladder.py" >/dev/null 2>&1 \
  || fail k "cell_assemble failed for a spec-capable recipe with --spec off"
read -r on method <<<"$(python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))["spec"]
print(d["on"], d["method"])' "$tmp/specoff.json")"
[ "$on" = "False" ] || fail k "--spec off must record spec.on=false, got $on"
[ "$method" = "none" ] || fail k "--spec off must record spec.method=none, got $method"
python3 "$HERE/validate_artifact.py" "$tmp/specoff.json" >/dev/null \
  || fail k "a spec-off cell on a spec-capable recipe must validate:
$(python3 "$HERE/validate_artifact.py" "$tmp/specoff.json")"
ok k "qwen3.6-35b-a3b-fp8 on h200 with --spec off records method=none and validates"

# ── (j) teardown only touches the container this invocation created ─────────
# The stubs are PATH-shimmed rather than passed through $DOCKER so that a
# hardcoded `docker` anywhere in the chain is caught too. nvidia-smi is stubbed
# because otherwise preflight fails first and the serve stage never runs.
mkdir -p "$tmp/bin"
cat > "$tmp/bin/docker" <<'SH'
#!/usr/bin/env bash
# A Docker whose bookkeeping is a file. A DETACHED `run` puts the container
# into the running set BEFORE it answers, because that is the order the real
# one works in -- `docker run -d` creates the container and then prints its ID
# -- and DOCKER_RUN_BLOCK_S widens the gap between the two until a signal fits
# inside it. A foreground `docker run --rm` (the launcher's kernel check)
# outlives nothing and is recorded nowhere. `ps -aq --filter` answers from that
# set, honouring BOTH a `label=` filter and an exact `name=^...$` one, and
# stop/rm remove from it, so a test can ask the only question that matters
# after an interruption: is the container this cell created still there?
printf '%s\n' "$*" >> "$DOCKER_CALLS"
running="${DOCKER_RUNNING:-}"
# `docker image inspect` and `docker inspect` answer the same questions here;
# the sub-verb is dropped so one branch serves both.
if [ "${1:-}" = "image" ] && [ "${2:-}" = "inspect" ]; then shift; fi
# What the mutable TAG resolves to right now. A tag is a pointer: re-pointing
# this file mid-run is a rebuild under the same tag, which is what makes
# inspecting the tag after a run the wrong question.
tag_image() { [ -s "${DOCKER_TAG_IMAGE:-}" ] 2>/dev/null && cat "$DOCKER_TAG_IMAGE"; }
# "<image-id-or-tag> <digest> <revision>" per line: what each image says about
# ITSELF, so image A's answers stay distinguishable from image B's.
registry() {  # registry TARGET FIELD -> the field, or nothing
  { [ -n "${DOCKER_REGISTRY:-}" ] && [ -f "$DOCKER_REGISTRY" ]; } || return 1
  awk -v t="$1" -v f="$2" '$1 == t { print $f; found = 1 } END { exit !found }' \
    "$DOCKER_REGISTRY"
}
case "${1:-}" in
  run)
    if [ "${DOCKER_RUN_RC:-0}" != "0" ]; then
      echo "docker: Error response from daemon: Conflict. The container name is already in use." >&2
      exit "${DOCKER_RUN_RC}"
    fi
    label=""; name=""; detached=0; prev=""
    for a in "$@"; do
      case "$prev" in
        --label) label="$a" ;;
        --name) name="$a" ;;
      esac
      if [ "$a" = "-d" ]; then detached=1; fi
      prev="$a"
    done
    if [ "$detached" = "1" ]; then
      if [ -n "$running" ]; then
        printf '%s %s %s\n' "${DOCKER_FAKE_CID:?}" "$label" "$name" >> "$running"
      fi
      # The image the create RESOLVED the tag to, recorded at create time the
      # way the daemon does. `inspect {{.Image}}` answers from here.
      if [ -n "${DOCKER_CREATED_IMAGES:-}" ]; then
        printf '%s %s %s\n' "${DOCKER_FAKE_CID:?}" "$name" "$(tag_image)" \
          >> "$DOCKER_CREATED_IMAGES"
      fi
      if [ -n "${DOCKER_RUN_BLOCK_S:-}" ]; then sleep "$DOCKER_RUN_BLOCK_S"; fi
    fi
    echo "${DOCKER_FAKE_CID:?}"
    ;;
  ps)
    # A lookup that FAILED is not a lookup that found nothing. The marker file
    # named by DOCKER_PS_FAIL_ONCE is CONSUMED by the first `ps`, so exactly
    # one call fails the way a daemon hiccup does.
    if [ -n "${DOCKER_PS_FAIL_ONCE:-}" ] && [ -f "$DOCKER_PS_FAIL_ONCE" ]; then
      rm -f "$DOCKER_PS_FAIL_ONCE"
      echo "docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock." >&2
      exit 1
    fi
    want_label=""; want_name=""; prev=""
    for a in "$@"; do
      if [ "$prev" = "--filter" ]; then
        case "$a" in
          label=*) want_label="${a#label=}" ;;
          name=*) want_name="${a#name=}"; want_name="${want_name#^}"; want_name="${want_name%\$}" ;;
        esac
      fi
      prev="$a"
    done
    { [ -n "$running" ] && [ -f "$running" ]; } || exit 0
    while read -r cid lab nm; do
      if [ -n "$want_label" ] && [ "$lab" != "$want_label" ]; then continue; fi
      if [ -n "$want_name" ] && [ "$nm" != "$want_name" ]; then continue; fi
      echo "$cid"
    done < "$running"
    ;;
  inspect)
    # Three questions share this verb: does a container name exist (the
    # launcher's pre-create probe, where "yes" is a REFUSAL), what digest the
    # image that ran has, and what revision or version that image declares.
    # The go template says which one is being asked.
    fmt=""; target=""; prev=""
    for a in "$@"; do
      if [ "$prev" = "--format" ]; then fmt="$a"; fi
      prev="$a"; target="$a"
    done
    case "$fmt" in
      *RepoDigests*)
        digest="$(registry "$target" 2)" || digest="${DOCKER_FAKE_DIGEST:-}"
        [ -n "$digest" ] || exit 1
        echo "$target@$digest"
        ;;
      *image.revision*)
        registry "$target" 3 || echo "${DOCKER_FAKE_REVISION:-<no value>}"
        ;;
      *image.version*) echo "${DOCKER_FAKE_IMAGE_VERSION:-<no value>}" ;;
      # The reference the create was GIVEN, and the image it resolved to. The
      # second is the immutable one, and the only honest answer to "what ran".
      *Config.Image*) echo "${DOCKER_FAKE_IMAGE_REF:-$target}" ;;
      *.Image*)
        { [ -n "${DOCKER_CREATED_IMAGES:-}" ] && [ -f "$DOCKER_CREATED_IMAGES" ]; } || exit 1
        awk -v t="$target" '$1 == t || $2 == t { print $3; found = 1 } END { exit !found }' \
          "$DOCKER_CREATED_IMAGES"
        ;;
      *)
        { [ -n "$running" ] && [ -f "$running" ]; } || exit 1
        while read -r cid lab nm; do
          if [ "$nm" = "$target" ]; then echo "true"; exit 0; fi
        done < "$running"
        exit 1
        ;;
    esac
    ;;
  stop|rm)
    { [ -n "$running" ] && [ -f "$running" ]; } || exit 0
    kept=""
    while read -r cid lab nm; do
      if [ "$cid" != "${2:-}" ] && [ "$nm" != "${2:-}" ]; then kept="$kept$cid $lab $nm
"; fi
    done < "$running"
    printf '%s' "$kept" > "$running"
    ;;
esac
exit 0
SH
cat > "$tmp/bin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
cat <<'Q'
==============NVSMI LOG==============
Driver Version                            : 999.00
CUDA Version                              : 99.9
Attached GPUs                             : 1
GPU 00000000:01:00.0
    Product Name                          : Stub Hopper
Q
SH
chmod +x "$tmp/bin/docker" "$tmp/bin/nvidia-smi"
DIGEST="sha256:$(printf 'a%.0s' $(seq 64))"
CID="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
# The stub's running set: what `docker run` created and stop/rm have not yet
# taken away. A cleanup that misses is visible here as a leftover line.
export DOCKER_RUNNING="$tmp/docker.running"; : > "$DOCKER_RUNNING"
# "<cid> <name> <image-id>": what each create resolved its tag to, at the
# moment it created the container.
export DOCKER_CREATED_IMAGES="$tmp/docker.created-images"; : > "$DOCKER_CREATED_IMAGES"

# (j1) vllm_control.sh hands the container ID back, or reports that it made none.
export DOCKER_CALLS="$tmp/vc.calls"; : > "$DOCKER_CALLS"
# The image this create resolves. The reference vllm_control launches is
# <repo>@<digest> and so cannot drift, but the container's own resolved ID is
# what stays true once the container is gone -- so it is recorded too.
VC_IMAGE="sha256:$(printf 'f%.0s' $(seq 64))"
printf '%s\n' "$VC_IMAGE" > "$tmp/docker.tag-image"
out="$(DOCKER_FAKE_CID="$CID" PATH="$tmp/bin:$PATH" VLLM_IMAGE_DIGEST="$DIGEST" \
        DOCKER_TAG_IMAGE="$tmp/docker.tag-image" \
        VLLM_CONTAINER=atlas-campaign-selftest-1 \
        bash "$VC" nemotron-3-nano-fp8 h100 --spec off \
        --label atlas-campaign.run=selftest-1 --id-file "$tmp/vc.id" \
        --image-file "$tmp/vc.image" 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail j "vllm_control with a stub docker exited $rc:
$out"
[ "$(tail -1 <<<"$out")" = "container_id: $CID" ] \
  || fail j "the container ID must be the last line: $(tail -3 <<<"$out")"
[ "$(cat "$tmp/vc.id")" = "$CID" ] || fail j "--id-file must hold the container ID"
have "$(cat "$DOCKER_CALLS")" "--label atlas-campaign.run=selftest-1" \
  || fail j "the ownership label must reach docker run: $(cat "$DOCKER_CALLS")"
ok j "vllm_control returns the created container ID and labels it as this run's"

have "$(cat "$tmp/vc.image" 2>&1)" "id=$VC_IMAGE" \
  || fail j "--image-file must hold the image the created container resolved, got:
$(cat "$tmp/vc.image" 2>&1)"
rm -f "$tmp/docker.tag-image"
ok j "--image-file records what the created container is running, not the tag"

: > "$DOCKER_CALLS"; rm -f "$tmp/vc.id"
out="$(DOCKER_FAKE_CID="$CID" DOCKER_RUN_RC=125 PATH="$tmp/bin:$PATH" \
        VLLM_IMAGE_DIGEST="$DIGEST" VLLM_CONTAINER=atlas-campaign-selftest-2 \
        bash "$VC" nemotron-3-nano-fp8 h100 --spec off --id-file "$tmp/vc.id" 2>&1)"; rc=$?
[ $rc -eq 125 ] || fail j "a name conflict must propagate exit 125, got $rc:
$out"
[ -e "$tmp/vc.id" ] && fail j "a failed docker run must write no container ID"
ok j "a docker run name conflict exits 125 and records no container ID"

# (j2) the cell: a name conflict stops and removes nothing.
: > "$DOCKER_CALLS"
out="$(DOCKER_FAKE_CID="$CID" DOCKER_RUN_RC=125 PATH="$tmp/bin:$PATH" \
        VLLM_IMAGE_DIGEST="$DIGEST" \
        bash "$RUN" --engine vllm --model nemotron-3-nano-fp8 --sku h100 \
        --workload lat --concurrency 1 --spec off --think off \
        --out "$tmp/collide" --yes 2>&1)"; rc=$?
[ $rc -eq 1 ] || fail j "a cell whose serve stage failed must exit 1, got $rc:
$out"
touched="$(grep -E '^(stop|rm) ' "$DOCKER_CALLS" || true)"
[ -z "$touched" ] || fail j "a name conflict must stop and remove NOTHING, saw:
$touched"
ok j "docker run exit 125 -> no stop, no rm, nothing of another run is touched"

stage="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["failing_stage"])' \
          "$tmp/collide/artifact.json")"
[ "$stage" = "serve" ] || fail j "a name conflict must fail the serve stage, got $stage"
python3 "$HERE/validate_artifact.py" "$tmp/collide/artifact.json" >/dev/null \
  || fail j "the collision artifact must still validate:
$(python3 "$HERE/validate_artifact.py" "$tmp/collide/artifact.json")"
ok j "a name conflict is recorded as a serve-stage failure in a valid artifact"

names="$(grep '^run ' "$DOCKER_CALLS" | grep -o -- '--name [^ ]*' | sort -u)"
have "$names" "atlas-campaign-" || fail j "the container name must be campaign-scoped: $names"
: > "$DOCKER_CALLS"
DOCKER_FAKE_CID="$CID" DOCKER_RUN_RC=125 PATH="$tmp/bin:$PATH" \
  VLLM_IMAGE_DIGEST="$DIGEST" \
  bash "$RUN" --engine vllm --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/collide2" --yes >/dev/null 2>&1
names2="$(grep '^run ' "$DOCKER_CALLS" | grep -o -- '--name [^ ]*' | sort -u)"
[ "$names" != "$names2" ] || fail j "two invocations must not share a container name: $names"
ok j "each invocation names its container uniquely ($names vs $names2)"

# ── (l) a TERMINATED cell tears down what it owns and says what happened ────
# The stub server answers 503 forever, which is exactly what a loading engine
# looks like, so the runner is parked in its boot poll when the signal lands --
# the moment a container of its own exists and no gate has finished.
port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
export STUB_HITS="$tmp/loading.hits"; : > "$STUB_HITS"
cat > "$tmp/loading_stub.py" <<'PY'
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

BODY = b'{"status": "loading"}'


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        with open(os.environ["STUB_HITS"], "a") as fh:
            fh.write(self.path + "\n")
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
python3 "$tmp/loading_stub.py" "$port" & stub_pid=$!

# Poll for a condition rather than sleeping a guessed interval: a fixed sleep
# either flakes on a slow box or wastes the difference on a fast one.
wait_for() {  # wait_for SECONDS WHAT COMMAND...
  local deadline what
  deadline=$(( $(date +%s) + $1 )); shift
  what="$1"; shift
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if "$@"; then return 0; fi
    sleep 0.2
  done
  echo "timed out waiting for $what" >&2
  return 1
}

: > "$DOCKER_CALLS"
DOCKER_FAKE_CID="$CID" PATH="$tmp/bin:$PATH" VLLM_IMAGE_DIGEST="$DIGEST" \
  VLLM_PORT="$port" \
  bash "$RUN" --engine vllm --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/killed" --yes > "$tmp/killed.log" 2>&1 &
run_pid=$!

wait_for 120 "the cell's own docker run" grep -q '^run ' "$DOCKER_CALLS" \
  || fail l "the cell never created its container:
$(cat "$tmp/killed.log")"
wait_for 120 "the boot poll to reach the stub" test -s "$STUB_HITS" \
  || fail l "the cell never reached its boot gate:
$(cat "$tmp/killed.log")"

kill -TERM "$run_pid"
wait "$run_pid"; rc=$?
kill "$stub_pid" 2>/dev/null; wait "$stub_pid" 2>/dev/null
log="$(cat "$tmp/killed.log")"

[ $rc -eq 143 ] || fail l "a cell killed with SIGTERM must exit 143, got $rc:
$log"
ok l "SIGTERM during the boot gate exits 143, not a bare -15"

have "$(cat "$DOCKER_CALLS")" "stop $CID" \
  || fail l "the container this cell created was never stopped:
$(cat "$DOCKER_CALLS")"
have "$(cat "$DOCKER_CALLS")" "rm $CID" \
  || fail l "the container this cell created was never removed:
$(cat "$DOCKER_CALLS")"
stray="$(grep -E '^(stop|rm) ' "$DOCKER_CALLS" | grep -v -F -- "$CID" || true)"
[ -z "$stray" ] || fail l "a terminated cell must touch nothing but its own container, saw:
$stray"
ok l "SIGTERM stops and removes exactly the container this invocation created"

have "$log" "killing pid" \
  || fail l "the in-flight boot-gate child must be killed by pid:
$log"
ok l "the boot-gate child this cell started is killed by pid, not left polling"

art="$tmp/killed/artifact.json"
[ -f "$art" ] || fail l "a terminated cell must still write its artifact:
$log"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail l "the interruption artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
ok l "a terminated cell still writes an artifact, and it validates"

read -r verdict stage_name <<<"$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d["verdict"], d["failing_stage"])' "$art")"
[ "$verdict" = "NO-GO" ] || fail l "an interrupted cell is a NO-GO, got $verdict"
[ "$stage_name" = "boot" ] \
  || fail l "an interrupted cell must name the stage it was killed in, got $stage_name"
ok l "the artifact is NO-GO at 'boot' -- the stage that was in flight"

python3 -c '
import json, sys
notes = json.load(open(sys.argv[1]))["notes"]
assert "interrupted by SIGTERM" in notes, notes
assert "boot stage" in notes, notes' "$art" \
  || fail l "the notes must record the interruption, not just a failed stage:
$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"notes\"])" "$art")"
ok l "the notes say it was interrupted by SIGTERM rather than that boot failed on its own"

# ── (m) SIGTERM while the container is still being created ──────────────────
# The narrower window than (l): `docker run -d` has made the container and has
# not yet printed its ID, so the ID this cell tears down by does not exist in
# this shell yet. The stub blocks in exactly that gap. Ownership therefore has
# to be recoverable from something chosen and recorded BEFORE the create -- the
# per-invocation label -- or an interrupted create leaks a live server.
: > "$DOCKER_CALLS"; : > "$DOCKER_RUNNING"
DOCKER_FAKE_CID="$CID" DOCKER_RUN_BLOCK_S=60 PATH="$tmp/bin:$PATH" \
  VLLM_IMAGE_DIGEST="$DIGEST" \
  bash "$RUN" --engine vllm --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/creating" --yes > "$tmp/creating.log" 2>&1 &
run_pid=$!

wait_for 120 "docker run to create the container" test -s "$DOCKER_RUNNING" \
  || fail m "the cell never got as far as creating its container:
$(cat "$tmp/creating.log")"

kill -TERM "$run_pid"
wait "$run_pid"; rc=$?
log="$(cat "$tmp/creating.log")"

[ $rc -eq 143 ] || fail m "a cell killed mid-create must exit 143, got $rc:
$log"
ok m "SIGTERM during the create exits 143"

[ -s "$DOCKER_RUNNING" ] && fail m "the container this cell created is still running:
$(cat "$DOCKER_RUNNING")
docker calls:
$(cat "$DOCKER_CALLS")"
ok m "the container the create had already made is gone, not leaked"

have "$(cat "$DOCKER_CALLS")" "stop $CID" \
  || fail m "the interrupted create's container was never stopped:
$(cat "$DOCKER_CALLS")"
have "$(cat "$DOCKER_CALLS")" "rm $CID" \
  || fail m "the interrupted create's container was never removed:
$(cat "$DOCKER_CALLS")"
stray="$(grep -E '^(stop|rm) ' "$DOCKER_CALLS" | grep -v -F -- "$CID" || true)"
[ -z "$stray" ] || fail m "the reconciliation must touch nothing else, saw:
$stray"
ok m "SIGTERM mid-create stops and removes exactly the container that was made"

art="$tmp/creating/artifact.json"
[ -f "$art" ] || fail m "a cell interrupted mid-create must still write its artifact:
$log"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail m "the mid-create interruption artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
read -r verdict stage_name <<<"$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d["verdict"], d["failing_stage"])' "$art")"
[ "$verdict" = "NO-GO" ] || fail m "an interrupted cell is a NO-GO, got $verdict"
[ "$stage_name" = "serve" ] \
  || fail m "a cell killed inside its create was killed at serve, got $stage_name"
ok m "the artifact is a valid NO-GO at 'serve'"

# ── (n) a REFUSED atlas launch owns nothing of a prior run's ────────────────
# The wrapper used to treat "I started the launcher" as "I own what this run
# directory records". The launcher, handed a directory that already held rank
# records and a port whose /health already answered, refused before creating
# anything -- and the wrapper's finalizer then ran --stop against that
# directory and killed the earlier run's rank. Nothing there was ever this
# invocation's, and the fix is that this invocation's run directory is its own.
prior="$tmp/prior-node-ep"
mkdir -p "$prior"
# A container record only: a pid record would make a failing test kill whatever
# process happens to hold that number on this box.
echo "prior-invocation-rank0" > "$prior/rank0.container"

ready_port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
cat > "$tmp/ready_stub.py" <<'READYPY'
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

BODY = b'{"status": "ready"}'


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
READYPY
python3 "$tmp/ready_stub.py" "$ready_port" >/dev/null 2>&1 & ready_pid=$!
wait_for 30 "the occupied /health endpoint" \
  curl -sf -o /dev/null "http://127.0.0.1:$ready_port/health" \
  || fail n "the stub endpoint never came up"

# The kernel audit cannot pass on a host with no GPU, and that is the stage
# failure this cell records. What matters is what the launcher does next.
cat > "$tmp/stub-spark" <<'STUBSH'
#!/usr/bin/env bash
echo "stub spark: $*"
echo "stub spark: no CUDA device on this host" >&2
exit 1
STUBSH
chmod +x "$tmp/stub-spark"

: > "$DOCKER_CALLS"; : > "$DOCKER_RUNNING"
out="$(PATH="$tmp/bin:$PATH" ATLAS_PORT="$ready_port" \
        ATLAS_NODE_RUN_DIR="$prior" SPARK_BIN="$tmp/stub-spark" \
        bash "$RUN" --engine atlas --model nemotron-3-nano-fp8 --sku h100 \
        --workload lat --concurrency 1 --spec off --think off \
        --out "$tmp/refused" --yes 2>&1)"; rc=$?
kill "$ready_pid" 2>/dev/null; wait "$ready_pid" 2>/dev/null

[ $rc -eq 1 ] || fail n "a cell whose serve stage failed must exit 1, got $rc:
$out"
have "$(cat "$tmp/refused/serve.log")" "REFUSED" \
  || fail n "the launcher was supposed to refuse the occupied endpoint:
$(cat "$tmp/refused/serve.log")"
ok n "the launcher refuses an occupied /health and the cell records a failed serve"

touched="$(grep -E '^(stop|rm) prior-invocation-rank0' "$DOCKER_CALLS" || true)"
[ -z "$touched" ] || fail n "a refused launch must stop nothing of the prior run, saw:
$touched"
[ -f "$prior/rank0.container" ] \
  || fail n "the prior run's own record was deleted by a launch that created nothing"
ok n "a refused launch leaves the prior run's rank and its record untouched"

art="$tmp/refused/artifact.json"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail n "the refusal artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
read -r verdict stage_name <<<"$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d["verdict"], d["failing_stage"])' "$art")"
[ "$verdict" = "NO-GO" ] || fail n "a cell that never served is a NO-GO, got $verdict"
[ "$stage_name" = "serve" ] || fail n "the failing stage must be serve, got $stage_name"
ok n "the artifact is a valid NO-GO at 'serve'"

# And the other half of the same rule: a launch that DID record its ranks is
# stopped. Ownership by evidence has to keep the case it was always right about.
cat > "$tmp/stub-spark-alive" <<'STUBSH'
#!/usr/bin/env bash
for a in "$@"; do
  if [ "$a" = "--check-kernels" ]; then echo "stub spark: kernels resolved"; exit 0; fi
done
echo "stub spark serve: $*"
sleep 45
STUBSH
chmod +x "$tmp/stub-spark-alive"

live_port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
: > "$STUB_HITS"
python3 "$tmp/loading_stub.py" "$live_port" >/dev/null 2>&1 & stub_pid=$!
PATH="$tmp/bin:$PATH" ATLAS_PORT="$live_port" SPARK_BIN="$tmp/stub-spark-alive" \
  bash "$RUN" --engine atlas --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/live" --yes > "$tmp/live.log" 2>&1 &
run_pid=$!

# The run directory carries the runner's own pid: that is what makes it this
# invocation's alone, and it is where the launcher's rank records land.
live_dir="$tmp/live/node-ep-$run_pid"
wait_for 120 "the launcher to record its rank" test -s "$live_dir/rank0.pid" \
  || fail n "no rank record appeared in $live_dir:
$(cat "$tmp/live.log")"
rank_pid="$(cat "$live_dir/rank0.pid")"

kill -TERM "$run_pid"
wait "$run_pid"; rc=$?
kill "$stub_pid" 2>/dev/null; wait "$stub_pid" 2>/dev/null
log="$(cat "$tmp/live.log")"

[ $rc -eq 143 ] || fail n "a terminated atlas cell must exit 143, got $rc:
$log"
# --stop reached this cell's OWN run directory and the record in it. Whether
# the rank was already down by then (the kill of the launcher's process tree
# can get there first) or --stop killed it is the platform's business; that the
# teardown went through the reserved directory and left the rank dead is not.
have "$log" "stopped 1 rank(s); logs kept in $live_dir" \
  || fail n "--stop never reached the run directory this cell reserved:
$log"
[ -e "$live_dir/rank0.pid" ] && fail n "the rank record this cell owned was not cleared:
$log"
kill -0 "$rank_pid" 2>/dev/null && { kill -9 "$rank_pid" 2>/dev/null; \
  fail n "rank pid $rank_pid survived the teardown"; }
ok n "a launch that recorded a rank in its own run dir is stopped and cleared"

art="$tmp/live/artifact.json"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail n "the interrupted atlas artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
ok n "the interrupted atlas cell still writes an artifact that validates"

# ── (o) SIGTERM inside an ATLAS container create ────────────────────────────
# The Atlas half of (m), and the same window: the launcher writes
# rank<N>.container only once `docker run -d` has returned, and the wrapper
# read an absent record as proof that nothing was created -- so a cell killed
# inside the create stopped nothing and left a rank holding the GPU. What
# survives that window is what the launcher wrote BEFORE the create:
# rank<N>.intent, carrying the deterministic container name and this launch's
# own label. The decoy below is why the reconciliation has to carry both: a
# container wearing the SAME name for a DIFFERENT run is somebody else's, and
# a name-only query would delete it.
atlas_port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
ACID="1111111111111111111111111111111111111111111111111111111111111111"
DECOY_CID="2222222222222222222222222222222222222222222222222222222222222222"
# What the image itself says it is, read back by (p) below: the digest
# `docker inspect` reports for it, and the revision label it carries.
ADIGEST="sha256:$(printf 'c%.0s' $(seq 64))"
AREV="4b825dc642cb6eb9a060e54bf8d69288fbee4904"
: > "$DOCKER_CALLS"; : > "$DOCKER_RUNNING"
DOCKER_FAKE_CID="$ACID" DOCKER_RUN_BLOCK_S=60 PATH="$tmp/bin:$PATH" \
  DOCKER_FAKE_DIGEST="$ADIGEST" DOCKER_FAKE_REVISION="$AREV" \
  IMAGE="avarok/atlas-fake:campaign-test" ATLAS_PORT="$atlas_port" \
  bash "$RUN" --engine atlas --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/creating-atlas" --yes > "$tmp/creating-atlas.log" 2>&1 &
run_pid=$!

wait_for 120 "the launcher's docker run to create rank 0" test -s "$DOCKER_RUNNING" \
  || fail o "the launcher never got as far as creating its container:
$(cat "$tmp/creating-atlas.log")"
created_name="$(awk 'NR==1 {print $NF}' "$DOCKER_RUNNING")"
[ -n "$created_name" ] || fail o "the create recorded no container name:
$(cat "$DOCKER_RUNNING")"
printf '%s %s %s\n' "$DECOY_CID" "atlas-node-ep.run=some-other-launch" "$created_name" \
  >> "$DOCKER_RUNNING"

kill -TERM "$run_pid"
wait "$run_pid"; rc=$?
log="$(cat "$tmp/creating-atlas.log")"

[ $rc -eq 143 ] || fail o "an atlas cell killed mid-create must exit 143, got $rc:
$log"
ok o "SIGTERM during an Atlas container create exits 143"

have "$(cat "$DOCKER_CALLS")" "stop $ACID" \
  || fail o "the rank the create had already made was never stopped:
$(cat "$DOCKER_CALLS")"
have "$(cat "$DOCKER_CALLS")" "rm $ACID" \
  || fail o "the rank the create had already made was never removed:
$(cat "$DOCKER_CALLS")"
ok o "SIGTERM mid-create stops and removes the rank the launcher had created"

left="$(awk '{print $1}' "$DOCKER_RUNNING" | tr '\n' ' ' | sed 's/ $//')"
[ "$left" = "$DECOY_CID" ] || fail o "the running set must hold the other run's container and
nothing else, holds: $left
docker calls:
$(cat "$DOCKER_CALLS")"
stray="$(grep -E "^(stop|rm) $DECOY_CID" "$DOCKER_CALLS" || true)"
[ -z "$stray" ] || fail o "a container of another run wearing the same name was touched:
$stray"
ok o "a same-named container from another run is neither stopped nor removed"

art="$tmp/creating-atlas/artifact.json"
[ -f "$art" ] || fail o "an atlas cell interrupted mid-create must still write its artifact:
$log"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail o "the mid-create interruption artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
read -r verdict stage_name <<<"$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d["verdict"], d["failing_stage"])' "$art")"
[ "$verdict" = "NO-GO" ] || fail o "an interrupted cell is a NO-GO, got $verdict"
case "$stage_name" in
  serve|boot) ;;
  *) fail o "a cell killed around its create was killed at serve or boot, got $stage_name" ;;
esac
ok o "the artifact is a valid NO-GO at '$stage_name'"

# ── (p) engine_version identifies the ENGINE, not the harness checkout ──────
# The reviewer's second finding: run_cell passed `git rev-parse HEAD` of this
# checkout as --git-sha for either engine and captured a digest only for vLLM,
# so an Atlas container cell reported the campaign's revision with
# image_digest=null and binary_sha256=null -- and validated as CERTIFIED. What
# an artifact has to name is the build that actually served the requests: the
# digest of the image that ran and the revision that image declares, or the
# hash of the binary that ran. The checkout SHA is harness provenance, and it
# now says so.
harness_sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"

atlas_art="$tmp/creating-atlas/artifact.json"
python3 - "$atlas_art" "$ADIGEST" "$AREV" "$harness_sha" <<'PY' || fail p "the Atlas container cell did not record the engine that ran: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["engine_version"], d.get("harness"))' "$atlas_art")"
import json, sys
doc = json.load(open(sys.argv[1]))
digest, rev, harness = sys.argv[2], sys.argv[3], sys.argv[4]
ev = doc["engine_version"]
assert ev["image_digest"] == digest, ev
assert ev["git_sha"] == rev, ev
assert ev["git_sha"] != harness, ev
assert ev["binary_sha256"] is None, ev
assert doc["harness"]["git_sha"] == (harness or None), (doc["harness"], harness)
PY
ok p "an Atlas container cell records the image's digest and revision, not the checkout"

local_art="$tmp/live/artifact.json"
python3 - "$local_art" "$tmp/stub-spark-alive" "$harness_sha" <<'PY' || fail p "the Atlas local-binary cell did not record the binary that ran: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["engine_version"], d.get("harness"))' "$local_art")"
import hashlib, json, sys
doc = json.load(open(sys.argv[1]))
ev = doc["engine_version"]
assert ev["binary_sha256"] == hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest(), ev
# `spark --version` prints CARGO_PKG_VERSION and nothing else
# (crates/spark-server/src/cli.rs), so there is no engine revision to record.
assert ev["git_sha"] is None, ev
assert ev["image_digest"] is None, ev
assert doc["harness"]["git_sha"] == (sys.argv[3] or None), doc["harness"]
PY
ok p "an Atlas local-binary cell hashes the binary it ran and claims no revision"

vllm_art="$tmp/creating/artifact.json"
python3 - "$vllm_art" "$DIGEST" "$harness_sha" <<'PY' || fail p "the vLLM cell's provenance changed: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["engine_version"], d.get("harness"))' "$vllm_art")"
import json, sys
doc = json.load(open(sys.argv[1]))
ev = doc["engine_version"]
assert ev["image_digest"] == sys.argv[2], ev
assert ev["git_sha"] is None, ev
assert ev["binary_sha256"] is None, ev
assert doc["harness"]["git_sha"] == (sys.argv[3] or None), doc["harness"]
PY
ok p "the vLLM cell keeps the digest it was pinned to and claims no engine revision"

# ── (r) the engine identity is the image that RAN, not the tag afterwards ───
# capture_engine_identity used to run AFTER teardown and inspect $IMAGE -- the
# tag. A tag is a mutable pointer: a rebuild or a pull between the launch and
# the finalizer re-points it, and the artifact then names a build that never
# served a request. What the container started from is its resolved image ID,
# recorded the moment the create returns and inspected before teardown removes
# the container. Here tag T resolves to image A at create time and is
# re-pointed to B during the run; the artifact must say A.
IMG_A="sha256:$(printf 'd%.0s' $(seq 64))"
IMG_B="sha256:$(printf 'e%.0s' $(seq 64))"
DIGEST_A="sha256:$(printf '1%.0s' $(seq 64))"
DIGEST_B="sha256:$(printf '2%.0s' $(seq 64))"
REV_A="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
REV_B="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
MUTABLE_TAG="avarok/atlas-fake:mutable"
export DOCKER_REGISTRY="$tmp/docker.registry"
export DOCKER_TAG_IMAGE="$tmp/docker.tag-image"
{
  printf '%s %s %s\n' "$IMG_A" "$DIGEST_A" "$REV_A"
  printf '%s %s %s\n' "$IMG_B" "$DIGEST_B" "$REV_B"
  # The TAG's own answers are B's, because by the time anything inspects the
  # tag it has been re-pointed. An artifact carrying these is the bug.
  printf '%s %s %s\n' "$MUTABLE_TAG" "$DIGEST_B" "$REV_B"
} > "$DOCKER_REGISTRY"
printf '%s\n' "$IMG_A" > "$DOCKER_TAG_IMAGE"

mutable_port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
: > "$STUB_HITS"
python3 "$tmp/loading_stub.py" "$mutable_port" >/dev/null 2>&1 & stub_pid=$!
: > "$DOCKER_CALLS"; : > "$DOCKER_RUNNING"; : > "$DOCKER_CREATED_IMAGES"
DOCKER_FAKE_CID="3333333333333333333333333333333333333333333333333333333333333333" \
  PATH="$tmp/bin:$PATH" IMAGE="$MUTABLE_TAG" ATLAS_PORT="$mutable_port" \
  bash "$RUN" --engine atlas --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/mutable-tag" --yes > "$tmp/mutable-tag.log" 2>&1 &
run_pid=$!

mutable_dir="$tmp/mutable-tag/node-ep-$run_pid"
wait_for 120 "the launcher to record the image its create resolved" \
  test -s "$mutable_dir/rank0.image" \
  || fail r "the launcher recorded no image for the rank it created:
$(cat "$tmp/mutable-tag.log")"
have "$(cat "$mutable_dir/rank0.image")" "id=$IMG_A" \
  || fail r "the launcher must record the resolved image ID, got:
$(cat "$mutable_dir/rank0.image")"
ok r "the launcher writes down the image its detached create actually resolved"

# The rebuild: same tag, different image, while the cell is still running.
printf '%s\n' "$IMG_B" > "$DOCKER_TAG_IMAGE"

kill -TERM "$run_pid"
wait "$run_pid"; rc=$?
kill "$stub_pid" 2>/dev/null; wait "$stub_pid" 2>/dev/null
log="$(cat "$tmp/mutable-tag.log")"

[ $rc -eq 143 ] || fail r "the interrupted cell must exit 143, got $rc:
$log"
art="$tmp/mutable-tag/artifact.json"
[ -f "$art" ] || fail r "an interrupted cell must still write its artifact:
$log"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail r "the artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
python3 - "$art" "$DIGEST_A" "$REV_A" "$DIGEST_B" "$REV_B" <<'RPY' \
  || fail r "the artifact names the build the tag points at NOW, not the one that ran: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["engine_version"])' "$art")"
import json, sys
ev = json.load(open(sys.argv[1]))["engine_version"]
ran_digest, ran_rev, rebuilt_digest, rebuilt_rev = sys.argv[2:6]
assert ev["image_digest"] == ran_digest, ev
assert ev["git_sha"] == ran_rev, ev
assert ev["image_digest"] != rebuilt_digest, ev
assert ev["git_sha"] != rebuilt_rev, ev
RPY
ok r "a tag re-pointed during the run does not change the build the artifact names"

grep -Fq -- "$IMG_A" "$DOCKER_CALLS" \
  || fail r "the identity lookup never inspected the recorded image ID:
$(cat "$DOCKER_CALLS")"
ok r "the identity is read from the recorded image ID, not from the tag"
unset DOCKER_REGISTRY DOCKER_TAG_IMAGE

# ── (q) a teardown lookup that FAILED is not a teardown that found nothing ──
# The launcher's half of this is scripts/start_node_ep_test.sh case (l); this
# is the runner's own label lookup. `docker ps --filter label=... || true`
# collapsed an unreachable daemon into an empty result, and the cell then
# printed "no container was created by this invocation" and finished its
# teardown stage cleanly -- while the container it created in the window the
# signal landed in kept the GPU. The artifact has to say the container may
# still be running; a silent clean teardown is the thing that loses it.
: > "$DOCKER_CALLS"; : > "$DOCKER_RUNNING"; : > "$tmp/ps-fail-once"
DOCKER_FAKE_CID="$CID" DOCKER_RUN_BLOCK_S=60 PATH="$tmp/bin:$PATH" \
  DOCKER_PS_FAIL_ONCE="$tmp/ps-fail-once" VLLM_IMAGE_DIGEST="$DIGEST" \
  bash "$RUN" --engine vllm --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/lookup-failed" --yes > "$tmp/lookup-failed.log" 2>&1 &
run_pid=$!

wait_for 120 "docker run to create the container" test -s "$DOCKER_RUNNING" \
  || fail q "the cell never got as far as creating its container:
$(cat "$tmp/lookup-failed.log")"

kill -TERM "$run_pid"
wait "$run_pid"; rc=$?
# The signal lands while the shell is inside `run_child ... > $OUT/serve.log`,
# and that redirection is the whole function call's -- so the finalizer the
# trap runs writes THERE, not to the runner's own log. Both are read.
log="$(cat "$tmp/lookup-failed.log" "$tmp/lookup-failed/serve.log" 2>&1)"

[ $rc -eq 143 ] || fail q "a cell killed mid-create must exit 143, got $rc:
$log"
stray="$(grep -E '^(stop|rm) ' "$DOCKER_CALLS" || true)"
[ -z "$stray" ] || fail q "the lookup answered nothing, so nothing may be stopped by guess:
$stray"
ok q "a teardown whose docker ps failed stops nothing on a guess"

have "$log" "may still be running" \
  || fail q "the teardown must say the container's fate is unknown, not that there was none:
$log"
grep -Fq "no container was created by this invocation" <<<"$log" \
  && fail q "an unreachable daemon was reported as proof that nothing was created:
$log"
ok q "the teardown reports an unknown fate rather than a confirmed absence"

art="$tmp/lookup-failed/artifact.json"
[ -f "$art" ] || fail q "the cell must still write its artifact:
$log"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail q "the artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
python3 -c '
import json, sys
doc = json.load(open(sys.argv[1]))
notes = doc["notes"]
assert "may still be running" in notes, notes
assert doc["verdict"] == "NO-GO", doc["verdict"]' "$art" \
  || fail q "the notes must carry the leaked container forward:
$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"notes\"])" "$art")"
ok q "the artifact validates as NO-GO and its notes name the container that may be leaked"

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
