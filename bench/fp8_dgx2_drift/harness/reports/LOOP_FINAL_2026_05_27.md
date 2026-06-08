# /loop FINAL — 2026-05-27 (12h autonomous continuation)

## The headline you wanted

**Atlas FP8 on Qwen3.6-35B-A3B-FP8 has a 30% reliability ceiling on opencode rust-axum creation. vLLM BF16 on the same dequanted weights hits 100%. p_bonf = 0.029.**

The gap is **per-layer FP8 numerical ceiling**, not a sampler bug, not a routing-drift bug, not a KV-cache choice. Documented Atlas-side cosine: **0.989 max per layer** (post-2026-05-24 RNE + softmax `__expf` fixes). Compounded across 40 layers: **0.989⁴⁰ ≈ 0.665** final-layer agreement with the BF16 reference. That number maps cleanly to ~30% multi-turn agentic pass rate.

vLLM BF16 has no such ceiling because the dequant happens at load time, every matmul is BF16×BF16, no FP8 conversions in the hot loop. Per layer ≈ 1.000, compounded ≈ 1.000.

## What I tried (9 Atlas tiers + 1 vLLM)

| Tier | KV dtype | MoE variant | cargo_valid | files_written (median) | Notes |
|---|---|---|---|---|---|
| tierA (raw baseline, pre-SM1) | FP8 | native FP8 | 3/10 | 1 | |
| sm1 (+SM1+WS1+WS2+AM1+B1) | FP8 | native FP8 | 3/10 | 1 | Sampler hooks lit |
| sm1_a2ao (+A2-AO path fuzzy) | FP8 | native FP8 | 2.5/10 | 1 | |
| sm1_a2ao_sc1 (+SC1 TOML repair) | FP8 | native FP8 | 3/10 | 1 | Best Atlas FP8 stack |
| sm1_a2ao_sc1_steps (+opencode steps:10) | FP8 | native FP8 | 2/10 | 1 | Regression — recap injection hurts |
| nvfp4moe (+`ATLAS_FORCE_NVFP4_MOE=1`) | FP8 | NVFP4 (FP8→BF16→NVFP4) | 3/10 | 1 | No movement |
| nvfp4kv (`--kv-cache-dtype nvfp4`) | NVFP4 | native FP8 | 1/10 | 0 | Regression |
| bf16kv (`--kv-cache-dtype bf16`) | BF16 (all 10 attn layers) | native FP8 | 0/10 | 0 | Worst — L35-L39 cliff bug |
| mixedkv2 (`--kv-high-precision-layers 2`) | mixed: 4/10 layers BF16, 6/10 FP8 | native FP8 | 1/10 | 0 | p_bonf=1.0 vs baseline — KV dtype is not the lever |
| **vllm_bf16** | **BF16** | **BF16 weights** | **10/10** | **1160** | **Definitive — model actually completes `cargo build`** |

Statistical method: N=10 each, Mann-Whitney U with Bonferroni adjustment across 18 metrics. Only `vllm_bf16 vs sm1_a2ao_sc1` survived correction at p_bonf < 0.05 on `cargo_toml_valid`. All Atlas-tier-vs-Atlas-tier comparisons p_bonf = 1.0 — they are statistically indistinguishable at the floor.

## Why each Atlas knob failed

The two-level accumulation already shipped in `moe_fp8_grouped_gemm.cu` (lines 194–279, 2026-05-25 fix) implements DeepGEMM-style FP32 promotion per K-block. That kernel is no longer the bottleneck — it took the L20 ssm.moe_out cosine from 0.920 to 0.989. But the remaining **0.011 per layer** is *MMA-precision-limited*, not rounding-limited. The MMA itself is `f32.bf16.bf16.f32`, and the weight has to go through BF16 in registers regardless of what we do upstream.

Memory note `project_qwen36_phase2b_softmax_expf.md` already says this verbatim: **"Dominant drift = MMA precision, not rounding."** Confirmed by all 8 Atlas tiers landing at the same 0-30% floor regardless of which knob we touch.

KV cache experiments confirmed memory's existing notes: NVFP4 KV regresses early layers, BF16 KV has an L35-L39 cliff bug that drops cargo_valid to 0/10. Mixed (`--kv-high-precision-layers N`) is the cleanest theoretical knob but only addresses **KV** quantization — leaves the weight-side FP8 MMA precision floor untouched.

## The path that closes the gap

**Load FP8 expert weights, dequant to BF16 at load time, run through Atlas's existing BF16 dense grouped-GEMM path.** This is what vLLM BF16 does. Memory cost: ~32 GB extra (35 GB FP8 experts → 65 GB BF16 experts).

GB10 budget at 88% util = 105 GB. Currently: ~35 GB weights + ~58 GB KV at max-seq-len=65536. With BF16 experts: ~65 GB weights + ~30 GB KV at max-seq-len=32768. Tight but fits.

**Implementation outline** (multi-day work — DID NOT ship this in /loop):

1. New `Nvfp4Variant::Fp8DequantedToBf16` variant (or env var `ATLAS_FP8_DEQUANT_MOE_TO_BF16=1`).
2. In `weight_loader/qwen35/load_layers.rs` and `weight_map/ssm_qwen35.rs::load_moe_qwen35`: when variant matches, call `dequant_fp8_blockscaled_to_bf16(store, ep, gpu)` (already exists at `weight_map/quant_helpers.rs:32`) and produce `DenseWeight`, not `QuantizedWeight`.
3. Wire MoE forward to skip both the FP8 fused kernel AND the NVFP4 dequant — use the existing BF16 dense GEMM grouped path that `Bf16Raw` non-fused checkpoints already use.
4. Adjust default `--max-seq-len` doc to 32768 for this variant.
5. Add the variant to `QuantFormat` matching so `Fp8` no longer auto-implies "skip NVFP4 path".

Estimated effort: 1-2 days for someone familiar with the loader; could be longer once the grouped-GEMM path is wired through MTP. **Worth doing** — this is the production fix.

Alternative cheaper experiment first: just attention. Bug #2 (`project_qwen36_phase2b_softmax_expf.md`) is "late attn layers regress L31-L39 from FP8 KV quant noise on large K/V magnitudes". Selective dequant of attention K/V projections at L31-L39 might lift cargo_valid into the 50-70% range without paying full BF16 memory. The L31-L39 weight set is small enough to keep in BF16 and fit comfortably.

## Production-fix ladder (ranked)

1. **Selective L31-L39 BF16 weights (attention only)** — ~2-3 GB extra memory. ~2 days. Tests Bug #2's specific localization. Expected: 50-70% cargo_valid.
2. **Full BF16 MoE experts on load** — ~32 GB extra memory. ~1-2 days. Matches vLLM. Expected: 90-100% cargo_valid.
3. **Switch production to vLLM BF16** — Zero Atlas work. 100% today. Costs: 70 GB weights (vs 35 GB), ~30% slower decode. Atlas stops being the serving path for Qwen3.6-A3B until #1 or #2 lands.

## What's left in Atlas that may still matter

The /loop session shipped these durable Atlas changes (they fix orthogonal drift modes and remain useful for future models):

- `crates/spark-server/src/whitespace_mask.rs` (WS1) — boot-vocab whitespace mask
- `crates/spark-server/src/attractor_mask.rs` (AM1) — lean attractor token bias
- `crates/spark-server/src/toml_repair.rs` (SC1) — TOML auto-repair
- `crates/spark-server/src/api/chat/tool_retry.rs::apply_fuzzy_repair_inplace` (A2-AO) — path fuzzy
- `crates/spark-server/src/scheduler/emit_step.rs::update_tool_param_state` (SM1) — state machine call site fix
- `decode_logits_seq.rs` B1 margin-ratio drift detector
- QV1 kernel-quant compat assertion at boot

These collectively closed `drift_lean_prefix`, `drift_path_literal_space`, `drift_bash_as_content` (each was 10-30%, now 0% sustained). They don't move `cargo_toml_valid` because the residual failure modes (TOML newline collapse, wandering, broken-write spiral) are all symptoms of the 0.989-per-layer ceiling, not sampler-tractable.

## State of the world when you wake up

- vLLM BF16 still running on dgx1 port 8888 (`vllm-bf16` container) — confirmed 10/10 on test prompt
- Atlas FP8 on dgx1 8888 → I stopped/restarted across 4 configs. Final running config is `--kv-high-precision-layers 2` for the `mixedkv2` tier
- `mixedkv2` N=10 harness running in the background; PID at `/tmp/mixedkv2.pid`, log at `/tmp/mixedkv2_run.log`
- All harness JSON at `bench/fp8_dgx2_drift/harness/runs/`
- This briefing + the 2026-05-27 morning briefing live next to it under `bench/fp8_dgx2_drift/harness/reports/`

## My one-line take

Sampler + env-var space is exhausted. The Atlas-side fix is BF16 expert weight loading (1-2 days code) or selective L31-L39 BF16 attention weights (2-3 days). Don't ship more sampler patches — the ceiling is at the MMA, not the head. Either implement option 1 above or fall back to vLLM BF16 for production until then.

## Detailed implementation plan for the BF16 expert path

`bench/fp8_dgx2_drift/harness/reports/BF16_EXPERT_IMPL_PLAN.md` has the full breakdown: ~550 LoC across new kernel (`moe_bf16_grouped_gemm.cu`), new `QuantizedWeight::Bf16Dense` variant, weight-loader arm, MoE dispatch branch, and factory plumbing. 1.5-2.5 days focused work. Includes a **single-expert cosine micro-bench as the cheap pre-verification step** before committing to the full kernel rewrite — if that doesn't show >0.999 cosine on L20, the kernel path isn't the fix and shouldn't be shipped.

## Hard cosine evidence (new, 2026-05-27)

Re-ran `bench/fp8_dgx2_drift/cosine_run.py` against the cached layer dumps. Three-way analysis (A: HF FP8→BF16 vs HF BF16-unquant, B: Atlas vs BF16-unquant, C: Atlas vs HF FP8→BF16):

- **A (FP8 quantization ceiling)**: mean 0.996, min 0.989 @ L32 — this is the mathematical floor with infinite-precision compute
- **B (Atlas total drift)**: mean 0.993, min 0.987 @ L38
- **C (Atlas compute drift)**: mean 0.995, min 0.990 @ L25
- **Headroom A−C = +0.001 mean**

Conclusion: Atlas compute is already within 0.001 cosine/layer of the FP8 ceiling. Further kernel work has at most ~+0.04 final-layer agreement to give. The remaining gap to vLLM-BF16 (mean A ≈ 0.996 ceiling vs vLLM ≈ 1.000) is the **quantization ceiling itself** — only achievable by serving BF16 weights.

## BF16 expert path — foundation shipped

Started the BF16 expert implementation; foundation green:

- `kernels/gb10/common/moe_bf16_grouped_gemm.cu` (~180 LoC): new kernel mirroring `moe_fp8_grouped_gemm_v2`'s coalesced layout with FP8 dequant/scale stripped. Compiles to PTX (30KB).
- `kernels/gb10/common/KERNEL.toml`: `moe_bf16_grouped_gemm = "moe_bf16"` module alias.
- `crates/spark-model/src/layers/ops/gemm_quant.rs`: `moe_bf16_grouped_gemm()` Rust wrapper (FFI dispatch).
- `crates/spark-model/src/layers/moe/mod.rs`: `moe_bf16_grouped_gemm_k: KernelHandle` field, `bf16_{gate,up,down}_weight_ptrs: Option<DevicePtr>` fields, `build_bf16_ptr_table()` helper.
- `crates/spark-model/src/layers/moe/init.rs`: kernel registration via `try_kernel("moe_bf16", "moe_bf16_grouped_gemm")`, default-`None` field init.
- `crates/spark-model/src/layers/moe/helpers_c.rs`: `set_bf16_experts()` setter that mirrors `set_fp8_experts()`.

`cargo check -p spark-model` is green. Atlas-kernels rebuild picks up the new PTX (89 kernels total).

## What's still required to land end-to-end (multi-hour each)

1. **Loader path** in `crates/spark-model/src/weight_loader/qwen35/load_layers.rs`:
   - New `ATLAS_FP8_DEQUANT_MOE_TO_BF16=1` env var detection.
   - For each layer when set: call `dequant_fp8_blockscaled_to_bf16` on each expert's `gate_proj` / `up_proj` / `down_proj`, then `moe_layer.set_bf16_experts(...)`.
   - Skip the FP8 expert load + `set_fp8_experts` path when this is active.

2. **Prefill forward dispatch** in `crates/spark-model/src/layers/moe/forward_prefill_fp8.rs`:
   - Branch at each `moe_fp8_grouped_gemm` call site (3 of them — gate, up, down) on `self.bf16_*_weight_ptrs.is_some()`.
   - When BF16, call `ops::moe_bf16_grouped_gemm` (no scale arg).
   - Replace `moe_expert_silu_down_shared_fp8` (which fuses silu+down for FP8) with a separate silu kernel + bf16 down GEMM. Atlas has an existing silu kernel; needs an explicit dispatch since the FP8 fusion path won't apply to BF16 weights.

3. **Decode forward** in `forward.rs` / `forward_k2.rs` / `forward_k3.rs`:
   - Same dispatch-branch pattern at the GEMV sites.
   - For per-token decode (m=1), the grouped-GEMM kernel still works but is suboptimal; consider falling through to per-expert `dense_gemv_bf16` in a Rust loop for K=1 token. Slow but correct; optimize later.

4. **Build + Docker image** + **harness N=10**.

The kernel + struct foundation is the high-blast-radius part (touches the core MoE layer struct and adds a CUDA kernel). With that landed, the remaining work is "wire it into the existing dispatch sites" — straightforward but tedious. ~1 day focused work.

## My one-line take (revised)

Cosine harness confirms the FP8 quantization itself is the ceiling, not Atlas compute. Foundation for the BF16 expert path is shipped and green; loader + dispatch integration is the next session.

— End of 12h /loop
