#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Sourced by campaign_test.sh; uses its temporary fixtures and assertions.
# shellcheck disable=SC2154,SC2034

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
[ "$verdict" = "NO-GO" ] || fail h "legacy think-off gates cannot certify think-on, got $verdict"
[ "$stage" = "coherency" ] || fail h "mismatched request policy must fail coherency, got $stage"
[ "$method" = "mean_of_rep_percentiles" ] \
  || fail h "a ladder-derived cell must not claim pooled percentiles, got $method"
ok h "think-on with legacy think-off gate evidence is NO-GO/coherency with mean_of_rep_percentiles"

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
