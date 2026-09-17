# DeepSeek-V4-Flash MTP Support — Implementation Plan

**Status:** Design complete & shape-grounded. Loader/forward to be implemented and runtime-verified once
`nvidia/DeepSeek-V4-Flash-NVFP4` is local on both nodes and the GPU is free.

**Why this model:** the RedHat `DeepSeek-V4-Flash-NVFP4-FP8` re-quant we ran has **zero** MTP weights.
`nvidia/DeepSeek-V4-Flash-NVFP4` ships a full MTP module (`num_nextn_predict_layers = 1`, 1575 tensors).
MTP is the highest-value remaining decode-throughput lever (memory-bound decode; a self-speculative draft
head amortizes the per-token weight traffic across accepted tokens).

---

## Key finding: the MTP module is a structural twin of a main V4 layer

Tensor shapes read from `model-00046-of-00046.safetensors` (hidden = 4096):

| Group | Tensors | Shape / dtype | Maps to |
|-------|---------|---------------|---------|
| **Combiner** | `mtp.0.enorm` | BF16 [4096] | RMSNorm on next-token embedding |
| | `mtp.0.hnorm` | BF16 [4096] | RMSNorm on previous hidden state |
| | `mtp.0.e_proj` | F8_E4M3 [4096,4096] + E8M0 scale | project normed embedding |
| | `mtp.0.h_proj` | F8_E4M3 [4096,4096] + E8M0 scale | project normed hidden |
| **Attn (MLA)** | `mtp.0.attn.wq_a` | F8 [1024,4096] | q down (q_lora=1024) — `MlaWeights` |
| | `mtp.0.attn.wq_b` | F8 [32768,1024] | q up (64 heads × 512) |
| | `mtp.0.attn.wkv` | F8 [512,4096] | kv down (kv_lora=512, MQA) |
| | `mtp.0.attn.wo_a/wo_b` | F8 [8192,4096]/[4096,8192] | o-proj low-rank |
| | `mtp.0.attn.q_norm/kv_norm` | BF16 [1024]/[512] | latent norms |
| | `mtp.0.attn.attn_sink` | F32 [64] | per-head sink. **No `compressor.*` → full attention** |
| **FFN (MoE)** | `mtp.0.ffn.gate.weight/.bias` | BF16 [256,4096] / F32 [256] | 256-expert router + noaux_tc bias (router stays BF16) |
| | `mtp.0.ffn.experts.E.w{1,2,3}` | I8 (NVFP4-packed) + E8M0 scale | identical to main-layer experts |
| **mHC** | `mtp.0.hc_attn_{base,fn,scale}` | F32 [24]/[24,16384] | `HcWeights` attn site (16384 = hc_mult 4 × 4096) |
| | `mtp.0.hc_ffn_{base,fn,scale}` | F32 | `HcWeights` ffn site |
| | `mtp.0.hc_head_{base,fn,scale}` | F32 [4]/[4,16384] | `HcHeadWeights` |
| **Head** | `mtp.0.norm` | BF16 [4096] | final norm before shared lm_head |

Everything except the 4 combiner tensors + final `norm` is **byte-identical in structure to a main V4
layer** that `assemble_layer()` already builds (MLA + mHC attn/ffn sites + hc_head + 256-expert NVFP4 MoE
with BF16 router). The embedding table and lm_head are **shared** with the main model (not duplicated in `mtp.*`).

### Combiner math (resolved from shapes — separate proj + sum, NOT concat)

```
e = embed(token)                       # shared embedding table  [4096]
h_in = e_proj( rmsnorm(e,  enorm) )    # [4096,4096] · [4096]
     + h_proj( rmsnorm(h_prev, hnorm)) # [4096,4096] · [4096]
```
(DeepSeek-V3 stored this as one [d,2d] `fc` on a concat; this checkpoint stores the two halves split as
`e_proj`/`h_proj`, which is algebraically the project-then-sum form above.)

### Per-draft forward
```
h_in = combiner(embed(prev_token), prev_hidden)   # above
h_out = v4_layer_forward(h_in)                     # REUSE: MLA + mHC(attn,ffn) + MoE/EP, full attention
logits = lm_head( rmsnorm(h_out, mtp.0.norm) )     # shared NVFP4 lm_head
draft  = argmax(logits)  (grammar-masked if active)
```

---

## What exists vs. what's needed

| Piece | State | Action |
|-------|-------|--------|
| Config (`num_nextn_predict_layers → num_mtp_modules`) | ✅ done | `config/parsers/deepseek_v4.rs:156-159` |
| Scheduler MTP step / verify / accept (`DraftProposer` trait) | ✅ done | `scheduler/mtp_step.rs`; V4 only needs to satisfy the trait |
| Generic Qwen `MtpHead` (standard-attn) | ❌ wrong vehicle | hardcodes q_proj/k_proj/v_proj + standard paged attn; no MLA, no mHC. Do **not** extend it. |
| V4 MTP weight loader | ✅ **done** | `weight_loader/deepseek_v4/mtp.rs` (`load_v4_mtp_module` → `DeepseekV4MtpModule`); reuses `assemble_layer` via a new `prefix` param. Compiles under `deny(warnings)`; main path byte-identical. |
| V4 MTP proposer (forward) | ⬜ to build | new `layers/deepseek_v4_mtp.rs` implementing `DraftProposer`, reusing `body.decode(...)` |
| Wire into model init | ⬜ to build | `model/impl_a1_init.rs` build path: when `num_mtp_modules>0` and `mtp.*` present, build the V4 proposer instead of the Qwen `MtpHead` |
| Crash-safety on the new tensors today | ✅ safe | loaders are by-name lookups → extra `mtp.*` tensors are ignored, current models unaffected |

### Loader — implemented (this commit)
- `assemble_layer(layer_idx, layer_prefix, …)` gained a `layer_prefix` arg. Main caller passes
  `layers.{i}` (byte-identical); the MTP loader passes `mtp.0` and `layer_idx = num_hidden_layers`
  so `compress_ratios.get()` / hash-layer / kv-dtype all default to (no compressor, no hash, bf16).
- `load_v4_mtp_module()` pre-loads the `mtp.0` MLA inputs (V4-Flash branch), loads its own
  `mtp.0.hc_head_*`, builds the body via `assemble_layer`, then loads the combiner
  (`enorm`,`hnorm`,`e_proj`,`h_proj`) + final `norm`. Returns `Ok(None)` when `num_mtp_modules==0`
  or no `mtp.0.*` tensors (safe no-op for the RedHat re-quant / non-MTP models).

### Proposer — next step, interface confirmed
Body invocation contract (from `layer/transformer_layer.rs` + `model/trait_impl/decode_a.rs:238`):
`body.decode(hidden, residual, state: &mut dyn LayerState, kv_cache, seq_len, block_table,
disk_block_ids, disk_last_offloaded_per_layer, ctx: &ForwardContext, stream)`.
`DeepseekV4MtpHead::propose()` will: combiner(`embed(last_token)`, `target_hidden`) → `body.decode`
into a dedicated single-slot MTP KV cache → `rmsnorm(_, norm)` → shared lm_head GEMV → (grammar-masked)
argmax; `after_verify()` trims the MTP KV slot. State (`LayerState` + MTP `PagedKvCache`) wraps in a
`ProposerState`. Each sub-step needs runtime validation against the real weights (mHC sinkhorn in a
single-token draft, EP all-reduce in the draft MoE, KV trim on rejection).

### Loader (fully specified — additive, no risk to main path)
1. Refactor `assemble_layer()` to take a `prefix: &str` (default `"layers.{idx}"`) so it can build the body
   from `"mtp.0"`. Pure parameterization — main-path behavior unchanged (verify byte-identical main load).
2. New `load_v4_mtp_module()`: call the parameterized body builder with `mtp.0`, then load the 5 combiner
   tensors (`enorm`,`hnorm`,`e_proj`+scale,`h_proj`+scale,`norm`). Note: expert scales here are named
   `.scale` (E8M0), confirm vs. main-layer `.weight_scale`/`.weight_scale_2` when shard 46 lands.
3. Store as `Option<DeepseekV4MtpModule>` on the model; gate all of it behind `num_mtp_modules>0`.

### Proposer (reuses every existing kernel — the runtime-verification-gated part)
Implement `DraftProposer` for `DeepseekV4MtpHead`:
- `propose()`: combiner → existing V4 layer decode forward (MLA decode + mHC sinkhorn + MoE EP all-reduce) →
  shared norm + lm_head → argmax. K=1 (1 module). MTP KV cache: one extra paged slot, same as main MLA.
- `after_verify()`: trim the MTP KV slot on rejection (mirror existing `MtpHead::after_verify`).
- **EP note:** the MTP MoE runs the same EP=2 all-reduce as main layers → the draft step does cross-node
  comm. Verify it composes with the scheduler's single-draft path (no CUDA-graph capture — that path
  deadlocks under EP on GB10, per the decode work).

---

## Verification plan (once model local + GPU free)

1. **Load:** server starts with `--speculative`, logs the V4 MTP module loaded (1575 tensors), no missing keys.
2. **Numerical:** per-layer cosine of the MTP body vs. a main-layer forward on the same input (the combiner +
   shared layer kernels are reused, so the body must match main-layer numerics within fp8 noise).
3. **Acceptance:** measure draft accept rate on coherent prose. DeepSeek-trained MTP heads typically hit
   ~80–90% (vs. the ~27% n-gram rate that made TRT-LLM spec-decode net-negative) — this is the whole point.
4. **Throughput gate:** decode tok/s MTP-on vs. MTP-off (baseline 15.5). Keep only if net-positive AND
   coherence + the prefill-vs-decode cosine guardrail hold. (Memory: K=2 MTP is net-negative on FP8 MoE-hybrid
   due to verify re-running the expert union; NVFP4 + a real trained 1-module head is the favorable case —
   measure, don't assume.)

## Open items to confirm against the real shard 46
- Expert scale tensor naming (`.scale` E8M0 here vs. `.weight_scale`/`_2` on main layers) — the loader must
  accept the MTP variant (mirror the earlier `weight_scale` vs `weight_scale_inv` loader fix).
- `wo_a`[8192,4096]/`wo_b`[4096,8192] o-proj rank (8192) vs. the main layer's o_lora_rank — confirm the
  grouped/block-diagonal o-proj path matches.
- mHC `hc_attn_fn`[24,16384] mix-dim (24) vs. main-layer hc site dims.

---

## Progress log (live)

**Loader — nvidia/DeepSeek-V4-Flash-NVFP4 now LOADS + SERVES EP=2** (was blocked):
- Fixed (committed): `I8` dtype; per-expert format dispatch (`load_expert_proj`): Standard NVFP4
  routed (`quantized`) + FP8 block-scaled shared experts (`quantized_from_fp8`); container rename
  `atlas-ds-ep*` (env reaper kills `atlas-deepseek-ep*`); start script → new model + explicit paths.
- Verified empirically via repeated EP=2 load tests: all 46 shards load, NCCL rendezvous OK, model
  builds, server answers at ~11.7 tok/s.

**BLOCKER: output is INCOHERENT** ("The three primary" → token salad). NOT the KV cache (fp8 AND
bf16 KV both garbage) → the weight dequant path is wrong for this checkpoint. Next: per-layer
prefill-vs-HF (or vs the known-good RedHat-format) cosine to localize which projection/expert
dequant diverges. Suspects: `quantized()` weight_scale_2 semantics (scalar vs 1/x), the FP8→NVFP4
shared-expert re-quant, or the NVFP4 routed `weight_scale` (E4M3 block) handling for this export.

**MTP routed experts** (`mtp.0.ffn.experts.*` = NVFP4 U8/I8 + E8M0 `.scale`) still bail — need an
E8M0-block-scaled NVFP4 GEMM path or a dequant-to-BF16/NVFP4 at load. Then the proposer.

Best image: `atlas-deepseek-v4:mtpload4` (loads+serves, incoherent). Branch deepseek-v4-clean.
