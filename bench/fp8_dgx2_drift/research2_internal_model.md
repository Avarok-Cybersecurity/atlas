# Atlas Qwen3.6 Forward-Pass Audit — Multi-Turn Coherence Bugs

Date: 2026-05-26
Scope: hot path code for Qwen3.6-35B-A3B-FP8 (10 full-attention layers, 30 GDN layers, MoE with 1 shared expert + N routed experts). Targets bugs that survive prefix caching, accumulate across turns, or interact with FP8 KV deep-layer drift / MTP K=2 rollback.

## Architecture refresher

Per-layer prefill flow (`crates/spark-model/src/layers/qwen3_ssm/trait_prefill.rs`, `trait_prefill_phase3.rs`):
1. `rms_norm_residual` (FP32-aware variant when `use_fp32_residual()`)
2. SSM in_proj (FP8 via `w8a16_gemm`), deinterleave, conv1d, GDN recurrence (WY32/WY4/persistent)
3. `gated_rms_norm_prefill` → out_proj (FP8 `w8a16_gemm`)
4. `residual_add_rms_norm` (FP32 hidden + BF16 out_proj_buf) → norm_output
5. MoE forward (`forward_prefill_fp8.rs`): shared expert via `w8a16_gemm`, routed experts via `moe_fp8_grouped_gemm` v2 (two-level FP32 accumulation, env-gated)
6. `residual_add` (FP32 hidden += BF16 moe_output)

State carried across requests: SSM `h_state` (FP32, per-layer per-slot), `conv_state` (FP32, per-layer per-slot), KV paged cache (FP8 E4M3 with per-attn-layer running k/v scales), MTP h_state checkpoint + intermediates for verify rollback, prefix cache + SSM snapshot pool keyed on session_hash.

## Findings

### F1 — `w8a16_gemm` is the BF16-truncation floor for ALL FP8 layers EXCEPT the routed MoE GEMM

`kernels/gb10/common/w8a16_gemm.cu:198-219` (and the transposed sibling `w8a16_gemm_t.cu:198-222`) dequantises every FP8 byte as `__float2bfloat16(E4M3_LUT[byte] * scale)` *before* the MMA. The 7-bit BF16 mantissa truncates `LUT × scale` to ~0.4 % per-weight error. The MoE routed grouped GEMM was specifically fixed to a DeepGEMM two-level accumulator (`moe_fp8_grouped_gemm.cu:194-280`) where `inner_acc` stays in FP32 across one K=128 scale block and the scale is applied ONCE per block to FP32 sums. None of the following call sites got the same treatment:

- Qwen3-attention QKV prefill (`qwen3_attention/prefill/paged_qkv.rs:144`, `cache_skip_qkv.rs:164`)
- Qwen3-attention O-proj prefill (`qwen3_attention/prefill/paged_oproj.rs:45`)
- SSM in_proj/out_proj (every dispatch in `qwen3_ssm/trait_prefill*.rs`; `out_proj` at `trait_prefill_phase3.rs:65-90`)
- MoE shared expert gate/up/down (`forward_prefill_fp8.rs:51-97`)
- MoE gate projection logits (`forward_prefill_fp8.rs:120-131`)

Each of these is hit *every prefill chunk and every decode step*, and the BF16 of the dequant value is then multiplied by an FP32-residual-quality activation. The compounding error is exactly the per-layer drift profile reported in Phase 2b (L31–L39 BF16-LUT cosine gap of 0.93–0.95). The MoE smoking-gun investigation already pinned 5/8 expert flips at L38 — but the shared-expert and attention paths share the same precision floor, which is why the deep-layer drift survives the MoE fix.

### F2 — FP8 KV mid-stream recalibration silently corrupts cached values

`crates/spark-model/src/layers/fp8_calibration.rs:138-207`. After the initial warmup freezes, `observe()` keeps re-firing every 128 tokens (`tokens_seen % 128 < num_tokens`) and EMA-blends the running absmax into `k_scale` / `v_scale` (lines 184-194). Single, layer-scoped scalars are used by both `inferspark_prefill_paged_fp8.cu` and `paged_decode_attn_fp8.cu` for the *entire* paged cache, so when the EMA shifts the scale the OLD K/V bytes (written under the previous scale) are now dequantised with the NEW scale. This is the "mid-prefill calibration → 0.92 cosine catastrophic regression" path identified in Phase 2b, but it is *still active in production at decode time*. Multi-turn workloads with topic shifts (which the comment explicitly justifies the recalibration on) deliberately move the absmax — every shift retroactively misreads tens of thousands of cached tokens. Net effect: long-running session cosine drift even when prefix caching is hot.

### F3 — Per-element BF16 scale × LUT in decode-fused FP8 MoE batch2 kernel

`kernels/gb10/common/moe_shared_expert_fused_fp8_batch2.cu:194-225`. The K=2 verify hot path uses `moe_expert_gate_up_shared_fp8_batch2`, which (correctly) keeps `acc` in FP32 but dequants every weight as `s_lut[byte] * sc` (line 217-225) where `sc` is `__bfloat162float` of a BF16 block scale. The FP32 product is then accumulated, so the MMA accumulator itself is fine. BUT this kernel handles *both* routed and shared experts for K=2 verify. For the shared expert (and the gate projection) this is functionally equivalent to F1's per-weight truncation, with the additional concern that the path is the dominant decode-time computation. The same kernel family also handles `silu_down_shared_fp8_batch2`, `gate_up_fp8_batch3`, and the K=2/K=3/single-token variants — every one of them mixes the per-element `s_lut[byte] * sc` pattern.

### F4 — Atomic ssm_layer_idx increment is non-symmetric in `commit_verify_state_async_dispatch`

`crates/spark-model/src/model/trait_impl/async_chkpt.rs:218-225`. When `commit_verify_state_async` runs partial-accept commit:

```rust
let Some(h_ckpt) = ssm.h_state_checkpoint else { ssm_layer_idx += 1; continue; };
let Some(conv_ckpt) = ssm.conv_state_checkpoint else { ssm_layer_idx += 1; continue; };
```

If `h_state_checkpoint` is `Some` but `conv_state_checkpoint` is `None`, `ssm_layer_idx` is bumped twice (once via the second `let-else`) and then again at the bottom of the loop (line 255). Today `alloc_sequence_dispatch` (`meta.rs:120-126`) always populates both Options together, but any future code that splits the allocation (e.g. lazy conv-state checkpoints behind a feature flag, or a transient state during slot copy in `sequence.rs:223-254`) would silently rotate intermediates onto the wrong layer. `start_rollback_and_checkpoint_async_dispatch` has the matching code shape but lacks the same defensive bail. CBD-class landmine for any future refactor.

### F5 — Prefix-cache hit with `marconi_skip` re-embeds last token only, leaving stale residual

`crates/spark-model/src/model/trait_impl/prefill_b/proc_range.rs:42-77`. When the prefix cache covers an entire chunk and it's the last chunk, the path re-embeds JUST the last token into `hidden[0]` (lines 59-69), then `prefill_b_forward_layers` is run with `proc_count=1` and `use_decode_path=true` (`forward_layers.rs:91`). That re-runs the per-layer decode path on a single embedded token, mutating the SSM `h_state` once more. But the residual stream buffer is *not* re-initialised: residual[0] still holds whatever the previous request's last layer left there. For Qwen3.6 with `use_fp32_residual()` the residual buffer is FP32 and `f32_residual_add` accumulates into it. The `rms_norm_residual` at the *start* of the first layer overwrites position 0 of `residual` (writes `hidden_after_add` back), so the leak is masked under normal conditions — but only if rms_norm_residual is actually the FP32 variant for this layer and writes back position 0 every layer. For Qwen3.6 layer 0 specifically (a GDN layer), `rms_norm_residual` is correctly the f32 variant. **Not currently a bug**, but the contract that prefix-cache reuse expects (residual gets clobbered before reading) is implicit and easy to break — flag this for the SSOT review.

### F6 — `intermediate[num_accepted-1]` rollback path is correct but conv-state checkpoint write order is racy under `--high-speed-swap`

`async_chkpt.rs:104-136`. The full-reject branch (num_accepted==0) restores `h_ckpt → h_state` then immediately writes `h_state → h_ckpt` again (lines 129-135) — this is a no-op write that exists only to keep the secondary stream synced; under high-speed-swap (`feedback_dgx1` & `project_high_speed_swap_phase62`) the slot pool can release the secondary stream's source buffer between the restore copy and the checkpoint copy. The `gpu.synchronize` at `sequence.rs:266` covers slot transitions but not the `(restore copy → checkpoint write-back)` pair on the secondary stream. Failure mode: rare cross-request SSM state pollution after an HSS evict + claim within a single verify cycle. Not the primary suspect for multi-turn drift but worth verifying with `start_rollback_*` event-ordering.

### F7 — `residual_add` post-MoE writes BF16 layer output into FP32 hidden each layer

`trait_prefill_phase3.rs:134-141` and `trait_prefill.rs:481-488`. With ATLAS_FP32_RESIDUAL=1, `hidden` is FP32 but `moe_output` is BF16. `f32_residual_add` correctly accumulates BF16 src into FP32 destination (`rms_norm.cu:734-743`), so the residual stream itself is fine. However, every per-layer contribution (MoE down_proj, attention out_proj, SSM out_proj) lands at BF16 just before the add — the FP32 residual benefit is "no truncation when summing 40 layers", not "no truncation per layer". For Qwen3.6 where hidden norms grow 18× from L0 to L38, the per-layer BF16 quantisation noise at deep layers is significant. Routes to F1: only the routed-MoE GEMM upgrades the final write to FP32-accumulated then cast.

### F8 — Softmax `sw_exp` was upgraded to `__expf` but the BF16 P*V truncation in prefill_paged_compute is still in place

`kernels/gb10/common/prefill_paged_compute.cuh:37-42`. P*V uses FP16 (`bf16x2_to_f16x2_bits`) for the probability tile, but the dot-product accumulator path is fed BF16 V values directly from smem after dequant. `sw_exp` is now `__expf` by default. The 10 full-attention layers (where the deep-L31–L39 drift sits) are the *only* ones using this kernel — so the precision is OK on the attention side. **Not a bug**, included for completeness.

## Ranked top-5 concrete bugs / precision-loss sites

1. **`kernels/gb10/common/w8a16_gemm.cu:213-217` and `w8a16_gemm_t.cu:214-217`** — Per-weight BF16-cast of `LUT × scale` before MMA. This is the *same* precision pattern that the MoE routed kernel had as the v1 bug. Affects shared-expert gate/up/down, MoE gate logits, all attention QKV/O-proj, all SSM in_proj/out_proj on Qwen3.6-FP8 and every other native-FP8 model. Drop-in fix: port the inner_acc/outer_acc DeepGEMM pattern from `moe_fp8_grouped_gemm.cu:194-280`. Expected gain: closes the L31–L39 BF16-LUT cosine gap reported in Phase 2b (0.93–0.95 → ≥0.97).

2. **`crates/spark-model/src/layers/fp8_calibration.rs:179-194`** — Periodic EMA recalibration of FP8 KV scales every 128 tokens after `frozen=true`. Mutates `k_scale`/`v_scale` mid-session, making the entire historical paged cache misread on every shift. Multi-turn topic switches deliberately trigger the path the code aims to optimise for, but the implementation is incompatible with a global per-layer scale. Fix: freeze scales hard after warmup OR switch to per-block FP8 KV scales (offline pre-prompt calibration pass per `project_qwen36_phase2b_softmax_expf.md` action item 3). Until then, set `ATLAS_FP8_KV_DISABLE_RECAL=1` if such an env exists, or gate `observe()` on `!frozen` only.

3. **`kernels/gb10/common/moe_shared_expert_fused_fp8_batch2.cu:217-225`** (and the family of `moe_shared_expert_fused_fp8_*` and `moe_expert_gate_up_shared_fp8_*` siblings) — Per-element `s_lut[byte] * sc` BF16-scale-quality dequant inside the decode-fused MoE kernel. K=2 verify hot path. Same DeepGEMM two-level accumulation pattern applies — extract the shared-expert + routed-expert contribution into FP32 accumulators per K=128 block.

4. **`crates/spark-model/src/model/trait_impl/async_chkpt.rs:218-225`** — Non-symmetric `ssm_layer_idx` increment when `h_state_checkpoint` and `conv_state_checkpoint` disagree on `Option<...>` populated-ness. Future-fragile; add `debug_assert` that both are Some-or-None together, OR refactor to a single Option<(h, conv)> and increment exactly once.

5. **`kernels/gb10/common/moe_fp8_grouped_gemm.cu:270` (and 438)** — `scale = __bfloat162float(S_exp[…])` reads BF16-stored per-block scales. The two-level accumulator correctly applies the scale to an FP32 inner sum, but the *scale value itself* is BF16-truncated at load. For 35B Qwen3.6's MoE the per-expert scale dynamic range is wide; promoting on-disk MoE block scales to FP32 (or doing a once-at-load conversion to FP32 device tensors) is a straight precision win. Pairs naturally with F1's port to the rest of the FP8 GEMMs.

## Cross-references

- `project_qwen36_phase2b_softmax_expf.md` — predicts that the dominant remaining drift is "FP8 MMA precision, not rounding"; F1 + F3 + F5 are concrete instances of MMA precision loss.
- `project_qwen36_moe_v2_fix.md` — the routed-MoE inner/outer fix is the template; F1, F3, F5 are the remaining sites that didn't get the same treatment.
- `project_qwen36_drift_moe_smoking_gun.md` — MoE expert routing flips 8/8 → 7/8 → 3/8 at L0/L24/L38 are gate-input drift driven; the gate projection itself uses `w8a16_gemm` (F1), so fixing F1 should sharpen gate decisions at deep layers.
- `project_mtp_k2_audit_2026_05_23.md` — Atlas's K=2 verify+commit was audited clean against vLLM #40880; F4 + F6 are landmines, not currently-active bugs.
- `project_chunked_prefill_root_cause.md` + `project_f74_chunked_prefill_fix.md` — upstream SSM chunked-prefill issues already mitigated by chunk-aligned splitting; F5 calls out the implicit residual-clobber contract on warm prefix-cache hits.

No code changes performed. All references are file-line citations against the current `/workspace/atlas-mtp` working tree.
