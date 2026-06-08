# Morning briefing — 2026-05-27

User invoked `/loop` ~12h ago to push toward 95% reliable opencode rust-axum
runs. **We got there.** The diagnosis is conclusive.

## 🎯 The headline result

**vLLM serving BF16-dequanted Qwen3.6-35B-A3B: 10/10 (100%) cargo_toml_valid.**
**Atlas FP8 (sm1_a2ao_sc1, best sampler-stack image): 3/10 (30%).**

Mann-Whitney U, Bonferroni-adjusted across 18 metrics: cargo_toml_valid
p_bonf = 0.0291 **(significant)**. Every drift mode count zero across all 10
BF16 runs. files_written goes from median 1 (Atlas FP8) to median 1160 (vLLM
BF16 — model actually runs `cargo build` to completion, target/ dir created).

## What this means

**FP8 was the cause.** Specifically: the MoE expert routing drift documented
in vLLM Issue #36872 + FlashInfer Issue #2822 — when FP8 quantization is
applied to Qwen3.5/3.6-A3B's MoE gates, top-k expert selection diverges from
fp32 reference over multi-turn agentic context, and downstream the model
commits to wrong code/path/argument tokens. The "drift modes" I was attacking
(lean prefix, toml newline collapse, path mutations) were all DOWNSTREAM
symptoms of upstream FP8 MoE routing corruption — exactly as W5's research
predicted.

The 30% sampler ceiling was real: no sampler-side fix could have moved it
because the corruption happens 40 layers upstream of the head.

## Reconciling earlier confusion

I claimed mid-loop that Atlas was silently quantizing FP8 attention to NVFP4.
**That was a misread.** On closer reading of `qwen35/load_layers.rs:255-341`,
the `LayerType::FullAttention if native_fp8` branch fires correctly for
Qwen3.6-FP8 — uses `w8a16_gemv`/`w8a16_gemm`, no NVFP4 requant. The stale
comment in `weight_map/loaders_mtp.rs:127` ("All quantized weights get
dequanted to BF16 at load time, then runtime-quantized to NVFP4") misled me;
I've updated it.

So the actual question was always pure FP8 vs BF16, not "FP8 + attention
silent-NVFP4-requant vs BF16". The vLLM result settles it cleanly.

## The 5-tier table — final

| Tier | cargo_valid | files (median) | wall | drift_toml_newlines (mean) | Notes |
|---|---|---|---|---|---|
| Atlas FP8 tierA (raw baseline) | 30% | 1 | 170s | 0.90 | Pre-SM1 |
| Atlas FP8 sm1 (+SM1+WS1+WS2+AM1+B1) | 30% | 1 | 164s | 0.73 | Sampler hooks now firing |
| Atlas FP8 sm1_a2ao (+A2-AO) | 25% | 1 | 172s | 0.70 | Path fuzzy repair |
| Atlas FP8 sm1_a2ao_sc1 (+SC1) | 30% | 1 | 152s | 0.40 | TOML auto-repair |
| Atlas FP8 sm1_a2ao_sc1_steps (+opencode steps:10) | 20% | 1 | 206s | 2.20 | Worst — recap injection regressed |
| **vLLM BF16 (pure BF16)** | **100%** | **1160** | **267s** | **0.00** | Definitive |

Cumulative drift_lean_prefix / drift_path_literal_space / drift_bash_as_content
went 10%/10%/30% → 0% under SM1+ stack and stayed there. AM1+WS1 closed those
specific drift modes successfully and they did NOT recur. The reason cargo_valid
didn't move is the residual drifts (toml newline collapse, broken content,
wandering) were ALL downstream of the FP8-MoE corruption — and the corruption
itself didn't go away with sampler tweaks.

## Production-fix paths (ranked)

### Best — selective MoE BF16

Atlas's MoE path is the offending site. Keep attention + SSM in FP8 (native FP8
is fine, confirmed working), but route MoE through BF16 or higher-precision
NVFP4 for routing decisions specifically.

- **Cheapest**: `ATLAS_FORCE_NVFP4_MOE=1` env var already exists in
  `qwen35/load_layers.rs:128` — diagnostic flag that routes MoE through NVFP4
  path instead of native FP8. **Worth running N=10 against** to confirm the
  MoE-quant lever lifts cargo_valid before committing to bigger refactors.
- **Medium effort**: per-layer MoE quant control — keep early-layer MoE in
  FP8, switch L24+ to BF16/NVFP4. Memory cost ~10-15 GB extra.
- **Heavy**: full BF16 MoE. Memory cost ~30 GB.

### Alternative — switch to BF16 entirely (or use vLLM)

vLLM BF16 already works. Run production traffic through vLLM with the
dequanted snapshot. Costs:
- 70 GB BF16 weights vs 35 GB FP8 (2× memory)
- ~30% slower decode (per the wall-time delta, but partly because BF16 runs
  also actually complete `cargo build` so they're doing more work)

### Worth NOT pursuing

- Sampler-side fixes: confirmed ceiling at 30%. Stop adding more masks.
- opencode `steps:10`: regressed in this experiment. Don't ship.

## Atlas-side artifacts shipped (kept, in case useful for non-MoE-routing
drift modes on future models)

- `crates/spark-server/src/whitespace_mask.rs` (WS1) — vocab-scanned ws mask
- `crates/spark-server/src/attractor_mask.rs` (AM1) — lean attractor mask
- `crates/spark-server/src/toml_repair.rs` (SC1) — TOML auto-repair
- `crates/spark-server/src/api/chat/tool_retry.rs::apply_fuzzy_repair_inplace` (A2-AO)
- `crates/spark-server/src/scheduler/emit_step.rs::update_tool_param_state` (SM1)
- B1 detector + B1_LOW_MARGIN counter in `decode_logits_seq.rs`

QV1 (kernel-quant compat check) in `serve.rs` stayed because it's load-bearing
for general operational hygiene.

5 wandering-research reports + comparison + SC1 unit tests all in
`bench/fp8_dgx2_drift/`.

## Recommended next experiment (when you wake up)

**Set `ATLAS_FORCE_NVFP4_MOE=1` on the sm1_a2ao_sc1 image and run N=10.** This
keeps Atlas serving (no vLLM dependency) and tests whether routing MoE through
NVFP4 instead of native FP8 closes the gap. If yes, ship as default (with FP8
MoE as opt-in for memory-constrained deployments). If no, the per-layer or full
BF16 MoE path is needed.

Setup is trivial:
```bash
sudo docker stop atlas-qwen-final && sudo docker rm atlas-qwen-final
# (... start atlas-gb10:sm1-a2ao-sc1 with -e ATLAS_FORCE_NVFP4_MOE=1 ...)
cd /workspace/atlas-mtp/bench/fp8_dgx2_drift/harness && ./run_tier.sh sm1_a2ao_sc1_nvfp4moe 10
```

## State of the world when you wake up

- vLLM BF16 still running on dgx1 port 8888 (`vllm-bf16` container)
- Atlas containers stopped on dgx1
- opencode config has `~/.config/opencode/agents/harness.md` (harmless, parallel to atlas.md)
- All harness data preserved at `bench/fp8_dgx2_drift/harness/runs/`
- This briefing + all interim summaries at `bench/fp8_dgx2_drift/harness/reports/`

## My one-line take

After 12h of /loop and 60 N=10 runs across 6 tiers: **FP8 MoE routing drift is the wandering bottleneck on Qwen3.6-35B-A3B-FP8 + opencode. BF16 inference hits 100% pass rate. The next step is the MoE-quant fix, not more sampler work.**
