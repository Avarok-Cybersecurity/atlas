# Arch Phase 02 — Prefill Op-by-Op Diff: Atlas vs vLLM (Qwen3.6-A3B-FP8)

**Model:** `Qwen3-Next-A3B-Instruct` (Qwen3.6) — 48 layers total: **36 GDN (linear-attention)** + **12 full-attention**.
**Quant target:** block-FP8 weights (E4M3, 1×128 weight groups), BF16 activations / KV cache.
**Scope:** Every op a single prefill pass executes on a contiguous batch of N prompt tokens, from `prepare_inputs` through layer-stack to logits. ~55 atomic ops walked.

Engine references:
- vLLM ‒ `/home/nologik/vllm/vllm/vllm/model_executor/models/qwen3_next.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/attention/backends/gdn_attn.py`, `vllm/model_executor/layers/{layernorm.py, activation.py, fused_moe/*}`.
- Atlas ‒ `crates/spark-model/src/layers/{qwen3_attention/prefill/, qwen3_ssm/, moe/}`, `crates/spark-model/src/model/trait_impl/prefill_b/`, `crates/spark-server/src/scheduler/prefill_a_step.rs`.

---

## TABLE 1 — Per-layer prefill op sequence (one decoder iteration)

Two sub-tables: 1A = full-attention layer (12 layers), 1B = GDN linear-attention layer (36 layers). MoE block at the tail is identical in both, so given once.

### 1A. Full-attention layer

| # | vLLM op | Atlas op | Fused? | Divergence / notes |
|---|---|---|---|---|
| 1 | `input_layernorm(x, residual)` → `GemmaRMSNorm.forward_native` (Python+`torch.compile`). Computes `residual += x; x = rmsnorm(residual)`. dtype: BF16/FP32-internal. (`qwen3_next.py:910`, `layernorm.py:331-346`) | `ops::rms_norm_residual` — fused CUDA kernel: `residual += hidden; out = rmsnorm(residual)`. dtype: BF16 io, FP32 reduce. (`prefill_inner.rs:59`) | **vLLM: relies on Inductor fusion via `@support_torch_compile`. Atlas: hand-fused.** | Both fused functionally. Atlas's kernel is one launch; vLLM materializes the Inductor-emitted Triton kernel on first run. Numerically equivalent (variance in FP32). |
| 2 | `QKVParallelLinear(x)` — **one fused QKV-with-gate GEMM** of `Q_size*2 + K_size + V_size`. FP8 block-scaled CUTLASS scaled-mm with **activation FP8 quant (per-token×128 group)** before MMA. (`qwen3_next.py:731-739`, `fp8_utils.py:305`) | **Three independent GEMMs**: Q‖gate, K, V via `prefill_attention_paged_qkv` → `ops::w8a16_gemm_t` ×3. Activations stay BF16; weights dequant FP8→BF16 in smem; MMA = `m16n8k16 bf16 bf16 f32`. (`paged_qkv.rs:42-89`, `w8a16_gemm_t.cu:128-217`) | vLLM: fused (1 GEMM). Atlas: not fused (3 GEMMs). | **CRITICAL — see arch_diff_02.** Atlas does W8A16, vLLM does W8A8 with per-token act-scales. Atlas pays 3× kernel launch + extra activation re-reads, AND has a 7-bit-mantissa round on every dequanted weight that vLLM does not. |
| 3 | `qkv.split([q*2, k, v])` → view+chunk; reshape Q‖gate per head, split → q, gate. View-only, no compute. (`qwen3_next.py:790-794`) | `deinterleave_qg_split_qnorm` — fused deinterleave + Q-norm in one kernel; OR `deinterleave_qg_split` if no q_norm. (`paged.rs:131-156`) | vLLM: view-only (free). Atlas: real kernel. | Atlas requires a separate kernel because its 3-GEMM path produced Q‖gate interleaved in `qkv_output`. vLLM's fused QKVParallel returns split-friendly contiguous chunks. |
| 4 | `q_norm(q_view)` — `GemmaRMSNorm` per-head, BF16. (`qwen3_next.py:798`) | Folded into op #3 (`deinterleave_qg_split_qnorm`) — Q-norm runs inside the deinterleave kernel. | vLLM: separate. Atlas: fused with deinterleave. | Atlas wins one launch and one global-memory round-trip on Q. |
| 5 | `k_norm(k_view)` — `GemmaRMSNorm` per-head, BF16. (`qwen3_next.py:801`) | `ops::rms_norm` on `k_contiguous`. (`paged.rs:223-235`) | Neither fuses K-norm with anything. | Equivalent. |
| 6 | `rotary_emb(positions, q, k)` — `get_rope(...)` dispatches to RoPE / YaRN / MRoPE. Applies in-place. (`qwen3_next.py:805`, `rotary_embedding.py`) | `ops::rope` / `rope_yarn` / `rope_mrope_interleaved` / `rope_proportional` selected by layer flags. (`paged.rs:293-386`) | Both: standalone kernel. | Equivalent math. Atlas has a B1 "fused K-path" optional kernel (`ATLAS_FUSED_KV=1`) that fuses k_norm+RoPE+BF16 write to KV cache; off by default. (`paged.rs:114-129, 416-440`) |
| 7 | KV cache write — implicit inside `Attention(q, k, v)` via attention-backend `reshape_and_cache_flash`. (`qwen3_next.py:807`) | `write_kv_cache` — explicit `reshape_and_cache_flash_*` kernel before attention. (`paged.rs:392-406`) | Both: standalone kernel. | Atlas writes first, then re-runs K through the optional fused kernel to overwrite the "triple-rounded" K — workaround for double-round drift specific to Atlas's BF16-MMA path (lines 413-440). vLLM does not need this. |
| 8 | Attention — `self.attn(q, k, v)` dispatched to `FlashAttention3` (Hopper/Blackwell), `FlashInfer`, or `xFormers` per backend selection. FA3 fuses softmax+output. BF16 io, FP32 accumulator. | `prefill_attention_with_cache_skip` (chunk 0) or `prefill_attention_paged_attn` (chunk 1+) → custom GB10 paged-flash kernel. (`paged.rs:443-494`) | Both: massively-fused single kernel. | Different kernels. vLLM's FA3 uses warp-specialized async memcpy + WGMMA; Atlas's GB10 paged-flash is hand-written. Numerical drift is in the softmax sum reduction (FP32 inside both). Atlas previously used a polynomial `sw_exp` (0.5% err) — replaced with `__expf` (project memory `project_qwen36_phase2b_softmax_expf.md`). |
| 9 | Attn-gate — if `attn_output_gate=True`: `gate = sigmoid(gate); attn_out *= gate`. Two element-wise ops, not fused. (`qwen3_next.py:809-811`) | `ops::sigmoid_gate_mul_batched` — single fused kernel. (`paged.rs:501-511`) | vLLM: 2 ops. Atlas: fused. | Atlas wins 1 kernel launch. |
| 10 | `o_proj(attn_gated)` — `RowParallelLinear`, FP8 block-scaled, TP allreduce inside. (`qwen3_next.py:813`) | `prefill_attention_paged_oproj` → `ops::w8a16_gemm_t` then explicit `comm.all_reduce_async`. (`paged.rs:515`, `prefill_inner.rs:157-177`) | vLLM: GEMM+AR coupled inside `RowParallelLinear`. Atlas: explicit separate AR. | Same W8A16-vs-W8A8 issue as op #2. |
| 11 | Optional `layer_scale` — element-wise `attn_out *= (1 + s)`. (`qwen3_next.py:928-936`) — **not present** on Qwen3.6. | (Same flag exists but inactive for this model.) | Both no-op. | — |
| 12 | `post_attention_layernorm(attn_out, residual)` — fused add+norm via `GemmaRMSNorm(x, residual)`. (`qwen3_next.py:939`) | `ops::residual_add_rms_norm` — fused. (`prefill_inner.rs:258-271`) | Both fused. | Same as op #1. |
| 13 | MLP/MoE — `Qwen3NextSparseMoeBlock.forward()`. (`qwen3_next.py:178-212`) | `self.ffn.forward_prefill(...)` — dispatches BF16/FP8 path. (`prefill_inner.rs:285-287`) | See **MoE sub-block** below. | — |
| 14 | Final residual add — implicit (vLLM passes `residual` forward; sum applied on next iteration's op #1). | `ops::residual_add` — explicit kernel. (`prefill_inner.rs:408-416`) | vLLM: deferred / folded into next iter's add-norm. Atlas: separate. | Equivalent math. Atlas costs one extra launch per layer; vLLM saves it via the running-residual carry. |

### 1B. GDN (linear-attention) layer

| # | vLLM op | Atlas op | Fused? | Divergence / notes |
|---|---|---|---|---|
| G1 | `input_layernorm(x, residual)` — same fused `GemmaRMSNorm`. | `ops::rms_norm_residual`. (`trait_prefill.rs:97-108`) | Both fused. | — |
| G2 | `in_proj_qkvz(x)` — one ColumnParallelLinear, FP8 block-scaled. (`qwen3_next.py:292`, dim = 2K+2V) | `prefill_qkvz_proj` → `ops::w8a16_gemm_t`. (`trait_prefill_proj.rs`) | Both: 1 GEMM. | W8A16 vs W8A8 same as 1A#2. |
| G3 | `fix_query_key_value_ordering(qkvz)` — Python reshape splits q/k/v/z. (`qwen3_next.py:454`) | Output of G2 is already deinterleaved `[N, qkvz_size=12288]`. | Both: view-only. | — |
| G4 | `in_proj_ba(x)` — separate ColumnParallelLinear (FP8 quant **disabled** because `ba_proj doesn't support blockwise fp8`, line 299). dim=2V. | Folded into next op (G5) — single fused kernel does BA-GEMM + gate compute. | vLLM: separate GEMM. Atlas: fused with G5. | Atlas wins one GEMM launch + a re-read of x. |
| G5 | `fused_gdn_gating(A_log, a, b, dt_bias)` → triton kernel computing `g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b)`. (`qwen3_next.py:598`, `qwen3_next.py:1339`) | `ops::dense_gemm_ba_gates_prefill` — **vectorized uint4 loads + warp-shuffle reduction + inline sigmoid/exp** in ONE kernel covering the BA GEMM AND gate compute. (`trait_prefill.rs:181-197`) | vLLM: 2 ops (GEMM + gate kernel). Atlas: 1 fused. | Atlas absorbs G4+G5 into a single kernel — clear Atlas win on launch overhead. |
| G6 | `mixed_qkv = cat([q,k,v], -1)` then transpose `(L,P) -> (P,L)`. View+`.transpose(0,1)`. (`qwen3_next.py:460`, `:564`) | Implicit — output of G2 is already `[N, conv_dim]` with q/k/v contiguous. | vLLM: cat+transpose. Atlas: layout-by-construction. | Atlas wins a layout op. |
| G7 | `causal_conv1d_fn(mixed_qkv_T, conv_weights, ...)` — Triton/FLA kernel; updates `conv_states` cache in place; **applies SiLU activation inside** (`activation=self.activation`). (`qwen3_next.py:567-577`) | `ops::conv1d_update_prefill` — custom CUDA kernel; updates conv state; **SiLU applied inside** as well. (`trait_prefill.rs:213-227`) | Both fused (conv + state-update + SiLU). | Equivalent. |
| G8 | `rearrange_mixed_qkv` — split conv_out into q,k,v slices. View-only. | Implicit pointer arithmetic: `q_ptr = conv_out`, `k_ptr = conv_out + key_dim`, `v_ptr = conv_out + 2*key_dim`. (`trait_prefill.rs:291-293`) | Both view-only. | — |
| G9 | L2-norm on q,k — **folded into `chunk_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)`** (`qwen3_next.py:654`). | `ops::l2_norm` — **separate** kernel before the delta-rule. (`trait_prefill.rs:253-263`) | vLLM: fused into delta-rule. Atlas: standalone. | Atlas takes one extra launch + global RW that vLLM does not. |
| G10 | `chunk_gated_delta_rule(q, k, v, g, beta, initial_state, ...)` — FLA Triton kernel doing the WY-recurrence in chunks; outputs both core_attn_out and last_recurrent_state. FP32 accum, BF16 state by default. (`qwen3_next.py:644-655`) | `prefill_gdn_recurrence` — dispatches to one of **WY4-persistent / WY32 / persistent / split4** custom kernels. FP32 accum, **FP32 state** (Atlas hard-codes). (`trait_prefill_recur.rs`) | Both heavily fused. | **Divergence: Atlas state is FP32, vLLM BF16 default.** Atlas safer. See arch_diff_10. |
| G11 | SSM state save: `ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(...)` — single index-write. (`qwen3_next.py:657-659`) | State written in-place by recurrence kernel; no separate copy. | — | Equivalent. |
| G12 | `RMSNormGated(core_attn_out, z, ...)` — `RMSNormGated.forward_cuda` dispatches to FLA's `rmsnorm_gated` triton kernel. norm + silu(z) gate fused. (`qwen3_next.py:486`, `layernorm.py:373-487`) | `ops::gated_rms_norm_prefill` — custom CUDA kernel; norm + gate fused, batched across all N tokens × heads. (`trait_prefill.rs:338-352`) | Both fused. | Equivalent. |
| G13 | `core_attn_out.reshape(...).rearrange("... h d -> ... (h d)")` — view-only. | Implicit — already correct layout. | Both view-only. | — |
| G14 | `out_proj(core_attn_out)` — `RowParallelLinear`, FP8. AR inside. (`qwen3_next.py:489`) | `prefill_out_proj_dispatch` (FP8 / NVFP4 / dense GEMM). (`trait_prefill.rs:378`) Then explicit AR? See note. | vLLM: GEMM+AR coupled. Atlas: GEMM separate, AR done at MoE level (op #M11). | — |
| G15 | `post_attention_layernorm(out_proj, residual)` — fused add+norm. | `ops::residual_add_rms_norm`. (`trait_prefill.rs:451-463`) | Both fused. | — |
| G16 | MLP/MoE — same `Qwen3NextSparseMoeBlock` as 1A#13. | `self.ffn.forward_prefill(...)` — same FP8 path. (`trait_prefill.rs:465-466`) | See MoE sub-block. | — |
| G17 | Final residual — same as 1A#14. | `ops::residual_add` — explicit. (`trait_prefill.rs:481-488`) | vLLM: deferred. Atlas: explicit. | — |

### 1C. MoE block (Qwen3NextSparseMoeBlock vs MoeLayer::forward_prefill_fp8)

| # | vLLM op | Atlas op | Fused? | Divergence |
|---|---|---|---|---|
| M1 | `gate(x)` — `ReplicatedLinear` (BF16, **not quantized**). Single small GEMM. (`qwen3_next.py:194`) | `ops::dense_gemm` on `weights.gate` (BF16). (`forward_prefill_fp8.rs:120-130`) | Both: 1 GEMM. | Equivalent. |
| M2 | `select_experts(gate_logits)` → `fused_topk` / `grouped_topk` (Triton kernel) — softmax+top-k+renormalize. (`fused_moe/layer.py:2155`, `fused_moe.py`) | `ops::moe_topk_softmax_batched` (or `moe_topk_sigmoid_batched` for sigmoid-routed models). (`forward_prefill_fp8.rs:141-167`) | Both fused. | Equivalent. |
| M3 | `moe_align_block_size` — Triton kernel computing per-expert token offsets + sorted token IDs. (`fused_moe.py:1893`) | `ops::moe_sort_by_expert` — does the same. (`forward_prefill_fp8.rs:178-190`) | Both kernels. | Equivalent. |
| M4a | **Shared expert**: `Qwen3NextMLP.forward` → `MergedColumnParallelLinear(gate_up_proj)` — **ONE fused FP8 GEMM producing gate‖up** (`qwen2_moe.py:86`). | **Two separate FP8 GEMMs**: `w8a16_gemm` on `gate_proj` and `up_proj` independently. (`forward_prefill_fp8.rs:51-74`) | vLLM: 1 GEMM. Atlas: 2 GEMMs. | **HIGH IMPACT** — vLLM saves 1 GEMM launch + 1 act-quant pass. |
| M4b | `silu_and_mul(gate_up)` — fused CUDA op `torch.ops._C.silu_and_mul`. (`activation.py:75`) | `ops::silu_mul`. (`forward_prefill_fp8.rs:76-84`) | Both fused. | Equivalent. |
| M4c | `down_proj(silu)` — single FP8 GEMM. | `ops::w8a16_gemm` on `down_proj`. (`forward_prefill_fp8.rs:86-97`) | Both: 1 GEMM. | W8A16 vs W8A8 again. |
| M5 | `experts(hidden, router_logits)` — `FusedMoE.forward_cuda` → `fused_experts` modular kernel: **one grouped W8A8-FP8 GEMM (gate+up fused via MergedColumnParallelLinear-equivalent expert weights), SiLU+mul, one grouped down GEMM, weighted reduce.** (`fused_moe/fused_moe.py:1897-1953`) | `ops::moe_fp8_grouped_gemm` ×3: separate gate, separate up, then act-mul, then down. Permute/unpermute explicit. (`forward_prefill_fp8.rs:238-298`) | vLLM: 2 grouped GEMMs (gate_up fused, down). Atlas: 3 grouped GEMMs. | **Same gate/up fusion gap as M4a.** Twice the launches + one extra grouped-GEMM trip through global memory for the routed experts. |
| M6 | Weighted reduce + unpermute back to token order — inside `fused_experts`. | `ops::moe_unpermute_reduce_indexed`. (`forward_prefill_fp8.rs:302-313`) | Both fused. | Equivalent. |
| M7 | TP/EP all-reduce on routed output — folded into `experts.maybe_all_reduce_tensor_model_parallel`. (`qwen3_next.py:208-210`) | `comm.all_reduce_async` explicit. (`forward_prefill_fp8.rs:324-328`) | — | Equivalent. |
| M8 | Shared blend: `routed + shared` — element-wise add (cheap; vLLM does `final_hidden_states[0] + final_hidden_states[1]`). (`qwen3_next.py:200`) | `ops::moe_batched_blend` — fused multiply-by-`sigmoid(shared_gate)` + add. (`forward_prefill_fp8.rs:343-353`) | Both small. | Atlas's blend kernel additionally applies a learned `shared_expert_gate` scaling; vLLM does this inside `Qwen3NextMLP(... expert_gate=...)`. |

---

## TABLE 2 — Cross-layer / batch-level ops

| Concern | vLLM | Atlas | Divergence |
|---|---|---|---|
| **Scheduler / chunked prefill** | `vllm/v1/core/sched/scheduler.py:Scheduler` — chunked prefill enabled by default; chunk size = `long_prefill_token_threshold` (default 8192). Hybrid models pin chunk to multiples of **mamba_block_size** so SSM state alignment is preserved (no explicit `_mamba_block_aligned_split` symbol on this branch — alignment enforced upstream by HybridKVCacheManager block sizes). | `crates/spark-server/src/scheduler/prefill_a_step.rs:start_chunked_prefill` (cap from `chunk_size`, default ~4096); iterates `forward_layers` per chunk. **Atlas SSM state is updated in-place across chunks via `LayerState`, so all 36 GDN layers process tokens in order — no block alignment needed.** (`prefill_b/forward_layers.rs`) | vLLM's chunk-alignment constraint exists because mamba block size affects KV-cache page allocation. Atlas threads SSM state through stateful `LayerState`, decoupling chunk size from state geometry. |
| **Prefix-cache fast-path** | Hybrid KV-cache manager: prefix hits free pages from the radix; SSM state for hybrid models snapshotted via `MambaCacheManager`. Effective: skip prefix compute, only run tail tokens through the model. | `prefill_b/prefix_lookup.rs:prefill_b_prefix_lookup` — Marconi radix lookup; if hit, restore `ssm_snapshot` (full per-layer FP32 h_state + conv_state) into `LayerState`, set `marconi_skip=true`, `kv_write_start=matched_tokens`. SSM-prefix lookup is **Atlas-specific** (Marconi). | **Atlas: explicit FP32 SSM snapshot save/restore for prefix-cache hits.** vLLM has a roughly equivalent `MambaCacheManager` (BF16 state). Marconi adds tree-shaped snapshots whereas vLLM is flat per-request. |
| **SSM snapshot save** | `gpu_model_runner.py` saves last state after each forward step; one snapshot per request. | `prefill_b/save_checkpoint.rs:prefill_b_save_checkpoint` — periodic intermediate snapshots every `ssm_checkpoint_interval` tokens (configurable); allows partial-prefix-cache restart at intermediate offsets. | **Atlas-specific.** Marconi tree allows tree-shaped prefix sharing (e.g., shared system prompt branching into different user turns). |
| **Position-id generation** | `gpu_model_runner._prepare_inputs` builds positions on CPU then `.to(device, non_blocking=True)`. For MRoPE multimodal: 3-stream (T,H,W) positions packed into one tensor. | `prefill_b_step.rs` builds positions; `meta_base` is one device tensor; MRoPE uses 3 strided offsets (`pos_stream_bytes`). (`forward_layers.rs:52-59`) | Equivalent. Atlas's `pos_stream_bytes` aliases a single allocation; vLLM uses one tensor of size `[3,N]`. **Bug fixed in Atlas:** previous `cuMemcpyHtoDAsync status 1` from MRoPE buffer sized for 1× instead of 3× (project_mrope_scratch_fix). |
| **Attention metadata prep** | `_prepare_inputs` builds `block_table`, `seq_lens`, `query_start_loc`, `slot_mapping`, `num_computed_tokens` on device. Per-backend `metadata_builder` then derives FA3-specific fields. For GDN: `GDNAttentionMetadataBuilder.build()` splits decode/prefill ranges (`gdn_attn.py:170` `split_decodes_and_prefills`). | `AttnMetadataDev { positions, slot, seq_len, block_table, max_blocks_per_seq, num_seqs }` (`forward_layers.rs:60-69`). Single struct, fewer fields. Decode-vs-prefill split lives at the **scheduler step level** (`prefill_b_step` vs `decode_step`), never co-mixed in one forward call. | **DIVERGENCE:** vLLM continuous-batches decode + prefill in one forward pass; Atlas separates them at the step boundary. vLLM's per-layer kernels must therefore handle mixed-batch inputs (spec/non-spec split). Atlas's kernels are simpler but throughput loses to vLLM when decode+prefill arrive together. |
| **Embedding lookup** | `VocabParallelEmbedding` — table lookup with TP shard. (`qwen3_next.py:982-988`) | `prefill_b/embed_chunk.rs` — `ops::embed_lookup`. | Equivalent. |
| **Final RMS-norm + LM-head** | `Qwen3NextRMSNorm` (`qwen3_next.py:999`) then `ParallelLMHead` GEMM → logits. | `prefill_b/finalize_last.rs` — last-token gather + `rms_norm` + LM-head GEMM. **Atlas only computes logits for the LAST token** of the prefill (one row); vLLM computes for all `num_logits_for_sampling` tokens which the sampler defaults to last token too. | Equivalent functionally. |
| **CUDA graphs** | `@support_torch_compile` on `Qwen3NextModel` (`qwen3_next.py:959`). Piecewise CUDA graph capture for prefill is supported in vLLM v1 (`compilation/decorators.py`). Inductor fusions applied first; graph captures the result. | **Atlas does NOT capture CUDA graphs during prefill.** Every prefill path sets `graph_capture: false` (`prefill_b/forward_layers.rs:84`, `prefill_b/batch_kernel.rs:399`). CUDA graphs are only used in **decode** (`decode_a.rs:144-285`). | **MAJOR DIVERGENCE.** vLLM amortizes kernel-launch overhead via piecewise CUDA graphs in prefill too. On Atlas, every prefill kernel is an explicit `cuLaunchKernel` — a measurable cost at ≥30 ops/layer × 48 layers = ~1440 launches per prefill. |
| **Inductor / `torch.compile`** | Yes — `@support_torch_compile` causes Inductor to fuse `add_rms_norm`, residual carries, activation+mul, and small element-wise chains into Triton kernels. `GemmaRMSNorm.forward_cuda` explicitly **torch.compiles** `forward_static`. (`layernorm.py:361-368`) | None — Atlas is pre-compiled Rust + hand-written CUDA. All fusions are explicit kernels (`rms_norm_residual`, `deinterleave_qg_split_qnorm`, `dense_gemm_ba_gates_prefill`, `sigmoid_gate_mul_batched`, `fused_k_norm_rope_cache_write_bf16_mrope`, `gated_rms_norm_prefill`). | Both reach similar fusion topologies; vLLM auto-derived, Atlas hand-rolled. Atlas's hand-fused kernels are sometimes **wider** than what Inductor finds (e.g., `dense_gemm_ba_gates_prefill` fuses GEMM+gating — Inductor can't fuse a GEMM with downstream pointwise). |
| **TP all-reduce** | Inside `RowParallelLinear` (after o_proj, after MoE down). | Explicit `comm.all_reduce_async` after o_proj (`prefill_inner.rs:167`) and after MoE (`forward_prefill_fp8.rs:327`). | Equivalent. Atlas has stream-event sync wrappers; vLLM uses NCCL functional API. |
| **NCCL stream** | Default. | Explicit `prefill_stream` (separate from compute) — previously experimented with overlapping lazy-down-proj transpose; rolled back due to GB10 stream-contention overhead (see `project_tp_overhead_breakdown`). | Equivalent. |

---

## Explicit flag-checks

### vLLM `add_rms_norm` fused kernel — Atlas equivalent?

**Partial.** vLLM's `GemmaRMSNorm.forward_cuda` (used in Qwen3.6) is **NOT** the `_C.fused_add_rms_norm` CUDA op (that op exists for the plain `RMSNorm` class, lines 252/285 of `layernorm.py`). Instead, GemmaRMSNorm routes through `torch.compile`d Python and relies on Inductor to emit a fused Triton kernel. Functionally fused.

Atlas has a direct hand-written equivalent: `ops::rms_norm_residual` (input-norm) and `ops::residual_add_rms_norm` (post-attn norm). Math matches Gemma's `(x*w).to(orig_dtype)` ordering (project memory confirms the v_norm/ones-buffer pattern in `paged.rs:245-257`). **YES — full functional parity, with Atlas hand-fused vs vLLM Inductor-fused.**

### vLLM `torch.compile` / piecewise CUDA graph — Atlas uses graphs in prefill?

**NO.** Atlas explicitly passes `graph_capture: false` everywhere in the prefill code-path. CUDA graphs are decode-only. **This is the single biggest launch-overhead delta on Atlas's prefill.**

### vLLM Inductor fusion passes — Atlas hand-fused kernels?

Mostly equivalent topology, both reach roughly the same fusion boundaries. Atlas has a few cases where its hand-fused kernels reach **wider** than Inductor can (BA+gate, deinterleave+q_norm). vLLM has fusions Atlas lacks (the entire fused QKV-with-gate GEMM is a Linear-layer construction, not a fusion pass — vLLM "fuses" by building the layer differently).

### vLLM `flashinfer.norm.fused_add_rmsnorm` — Atlas?

vLLM Qwen3-Next does **not** route Gemma-style RMSNorm through `flashinfer.norm.fused_add_rmsnorm`; it routes through the `torch.compile`d Python (because GemmaRMSNorm's `(1+w)` and `(x*w).to(dt)` differ from FlashInfer's plain RMSNorm). So this op is not used on the Qwen3.6-A3B prefill path in vLLM, and is irrelevant to a head-to-head.

### Atlas-specific kernels NOT in vLLM

- **Marconi SSM snapshot save/restore** — tree-shaped FP32 h_state + conv_state snapshots keyed by token-radix; allows shared-prefix branching across requests. vLLM's `MambaCacheManager` is flat per-request.
- **`fused_k_norm_rope_cache_write_bf16_mrope`** — single-rounded K-path that bypasses Atlas's BF16-MMA double-round drift. Optional, off by default (`ATLAS_FUSED_KV=1`).
- **B1 raw-K save scratch** — copies raw K post-QKV-GEMM to scratch so the fused-K kernel can overwrite triple-rounded K in cache.
- **Defensive memset of expert buffers** before MoE grouped GEMMs (`forward_prefill_fp8.rs:228-235`) — workaround for stale data in `max_m_tiles`-padded rows.
- **TurboQuant turbo4 KV cache** (project_turboquant) — WHT + Lloyd-Max 4-bit KV path. Additive option, not on by default for FP8 path.
- **Periodic intermediate SSM checkpoints** during a single prefill (`ssm_checkpoint_interval`) — Marconi-specific.
- **`gated_rms_norm_prefill`** batched across all N×heads in one launch — vLLM's `RMSNormGated.forward_cuda` is a Triton kernel from FLA, similar but separate kernel.

### vLLM-specific kernels / passes NOT in Atlas

- **`@support_torch_compile` on `Qwen3NextModel`** — Inductor fuses Python-level pointwise chains automatically.
- **Piecewise CUDA graph capture** in prefill (vLLM v1 compilation infra).
- **`MergedColumnParallelLinear` for `gate_up_proj`** — single GEMM producing concatenated gate‖up for both shared and routed expert MLPs. Atlas does 2 separate GEMMs.
- **`QKVParallelLinear` with `attn_output_gate=True`** — single GEMM for Q+gate+K+V. Atlas does 3 separate GEMMs.
- **W8A8 block-FP8 activation quantization** (`per_token_group_quant_fp8` before every FP8 GEMM) — Atlas keeps activations BF16 (W8A16).
- **`flashinfer.norm.fused_add_rmsnorm`** — defined but routed around for GemmaRMSNorm; available for the plain RMSNorm path.
- **`use_qk_l2norm_in_kernel=True`** inside FLA `chunk_gated_delta_rule` — Atlas does L2-norm as a separate kernel before its delta-rule.
- **`fused_gdn_gating` Triton kernel** — Atlas folds gating into `dense_gemm_ba_gates_prefill` (BA-GEMM + gates). Atlas is more fused here.
- **Mixed-batch decode+prefill in one forward pass** (continuous batching) — vLLM's GDN attention metadata builder explicitly splits inside the kernel; Atlas separates at the scheduler step.

---

## Missing on Atlas

1. **Piecewise CUDA graphs in prefill.** vLLM's `@support_torch_compile` + Inductor + cudagraph_replay covers prefill. Atlas's prefill is fully eager. Estimated cost: 30+ kernel launches/layer × 48 layers × ~5-10 µs CPU-side launch ≈ 7-15 ms per prefill of pure overhead, which scales with batch×layers.
2. **Fused gate_up GEMM** (`MergedColumnParallelLinear`) — both for the shared expert MLP and for routed-expert grouped GEMMs. Atlas does 2 GEMMs where vLLM does 1.
3. **Fused QKV-with-gate GEMM** (`QKVParallelLinear(attn_output_gate=True)`) — Atlas does 3.
4. **W8A8 block-FP8 path** — Atlas's `w8a16_gemm_t` keeps activations in BF16 and rounds dequanted weights to BF16 before MMA. vLLM keeps E4M3 magnitude and applies block scales in FP32 inside scaled-mm. This is the **primary FP8-drift contributor** flagged across the existing arch_diff_* reports.
5. **L2-norm fused into delta-rule** (`use_qk_l2norm_in_kernel`) — Atlas runs L2 as a separate kernel.
6. **Continuous batching of decode + prefill in one forward** — Atlas separates them. Throughput cost on multi-tenant traffic.
7. **`torch.compile`-driven small-op fusion** for the rest of the model (where Atlas has not explicitly written a fused kernel).

## Missing on vLLM

1. **Marconi tree-shaped SSM snapshot system** with periodic intermediate checkpoints during a single prefill — vLLM has flat per-request `MambaCacheManager`.
2. **Wider hand-fused kernels** at a few specific points (`dense_gemm_ba_gates_prefill` fuses GEMM+gates which Inductor cannot; `deinterleave_qg_split_qnorm` fuses Q-deinterleave with Q-norm).
3. **TurboQuant turbo4** KV-cache 4-bit format (additive option, not default for FP8).
4. **Defensive zeroing of `max_m_tiles`-padded expert buffers** — though this is a workaround, not a feature.

---

## Likely impact assessment

**Performance (throughput / TTFT):**
- vLLM's three structural fusion advantages (#1 piecewise CUDA graphs in prefill, #2 fused QKV+gate, #3 fused gate_up) compound: an FP8 prefill on Qwen3.6-A3B almost certainly runs **1.3-1.8× faster in tok/s than Atlas at the same batch shape**. Project memory `project_fp8_ceiling_conclusive` (2026-05-27) records Atlas FP8 hitting only 30% opencode parity vs vLLM BF16 — most of that gap is *quality*, but the perf-headroom gap explained above is consistent with the underlying timing pattern.
- The **kernel-launch-overhead delta** from missing prefill CUDA graphs is shape-dependent; it dominates at small batch / short prefill, fades at long prefill where each kernel is GEMM-bound.

**Numerical drift (the actual purpose of this comparison):**
- **W8A16 (Atlas) vs W8A8 (vLLM)** is the single largest functional divergence at every FP8 GEMM (op #2 in 1A, G2 in 1B, M4a / M4c / M5 in MoE). Each Atlas FP8 GEMM rounds every dequanted weight to BF16 *and* downcasts the per-block weight-scale to BF16, before the MMA reads any of it. This is a per-element 7-bit-mantissa round on every weight on every prefill. The cumulative drift is the leading hypothesis for the deep-layer regression fingerprint (L31-L39, recorded in `project_qwen36_phase2b_softmax_expf.md`).
- **The 3-GEMM QKV split** has additional drift implications: each GEMM has its own act-quant noise distribution (well, "would have" if Atlas quantized activations) and accumulator. With Atlas at W8A16 this is mostly a perf cost, not a drift cost — but if Atlas ever moves to W8A8, three separate per-token scales × 3 GEMMs is strictly worse numerically than one fused QKV scale × 1 GEMM (vLLM).
- **L2-norm-outside-kernel** (Atlas) vs **L2-inside-kernel** (vLLM): functionally identical math, but Atlas's separate kernel writes intermediate FP32→BF16→FP32 through global memory, introducing one extra round-trip rounding step that vLLM avoids. Small drift contribution.
- **Atlas's FP32 SSM state** (vs vLLM's BF16 default) means Atlas is *less* drift-prone on the GDN side, not more. If Atlas drifts more than vLLM on GDN layers, it's not state precision — it's the attention/projection path.
- **Atlas's "B1 fused K-path"** (off by default) and the "raw-K-save scratch + re-overwrite" pattern document a known **K-side double-/triple-rounding bug** during chunk-1+ prefill that vLLM does not have. Enabling `ATLAS_FUSED_KV=1` closes this gap on BF16 KV cache.

**Bottom-line conclusion for the FP8 drift hunt:**

The structural prefill differences point to **three Atlas-specific drift channels** that vLLM does not have:
1. **BF16-rounded dequanted FP8 weights** (every FP8 GEMM, every layer).
2. **BF16-downcast weight scales** (loader-time precision loss, also every FP8 GEMM).
3. **Triple-rounded K** on chunk-1+ paged prefill when `ATLAS_FUSED_KV` is off.

Channels #1 and #2 are global and uniform; channel #3 is shape-dependent (only chunks past chunk 0). Combined with the masking effect documented in `project_qwen36_phase2b_softmax_expf` (the prior polynomial-sw_exp masked deep-layer FP8 KV noise), the current FP8-drift signal is consistent with #1+#2 dominating at every layer, plus a sharpening at deep layers from K/V magnitudes accumulating. None of the GDN-side ops shows a structural drift advantage for vLLM — Atlas's FP32 state actively helps. The drift war is being fought on the FP8 GEMM side, not the SSM side.
