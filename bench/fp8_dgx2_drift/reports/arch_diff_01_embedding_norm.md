# arch_diff_01: Embedding lookup + first RMSNorm — Atlas vs vLLM

Model: Qwen3.6-A3B (`Qwen3-Next-80B-A3B-Instruct-FP8-Dynamic`), GDN hybrid.
Scope: `input_ids → embed_tokens → input_layernorm(layer 0)`.

## 1. Embedding lookup

**vLLM** — `VocabParallelEmbedding.forward_native` (`vllm/model_executor/layers/vocab_parallel_embedding.py:461`)
`output_parallel = self.quant_method.embedding(self, masked_input.long())` →
`F.embedding(input_, layer.weight)` (`:71`). Weight created at line 282-308 with
`params_dtype = torch.get_default_dtype()` → BF16. Output is BF16 `[N, H]`.
TP=1 path skips the all-reduce. Qwen3-Next does **no** embedding scaling
(`qwen3_next.py:1003-1004`: bare `return self.embed_tokens(input_ids)`).

**Atlas** — `prefill_b_embed_chunk_at` (`crates/spark-model/src/model/trait_impl/prefill_b/embed_chunk.rs:31-87`).
Token IDs uploaded via `copy_h2d_async`, then `ops::batched_embed` →
`batched_embed` kernel (`kernels/gb10/common/embed_from_argmax.cu:41-54`):
straight BF16 gather, output BF16 `[N, H]`. `scale_embeddings` (`impl_a3.rs:67-78`)
is dispatched but is a **no-op** for Qwen3.6 because
`config.embed_scale == 0.0` (default; only set by Gemma-4 parser).

With `ATLAS_FP32_RESIDUAL=1` (active per `MISSION_PROGRESS.md:386`), Atlas swaps
to `batched_embed_f32` (`embed_from_argmax.cu:57-70`, `impl_a1.rs:86-90`): reads
BF16 embed_table, upcasts to FP32 in the output. vLLM has **no equivalent
FP32-output embed path**.

Dtype summary: **vLLM BF16; Atlas BF16 (default) or FP32 (FP32-residual flag)**.

## 2. Residual stream dtype

**vLLM** is BF16 end-to-end for Qwen3-Next: at layer 0 `qwen3_next.py:906-907`
`residual = hidden_states` aliases the raw BF16 embedding; subsequent layers
read/write `residual` as BF16 through `GemmaRMSNorm.forward_static` (the
residual-add branch at `layernorm.py:331-337` keeps BF16 unless dtype is FP16).

**Atlas** with `ATLAS_FP32_RESIDUAL=1` upgrades the residual buffer to FP32
(`atlas-core/src/config/methods.rs:236-265`). This is a **deliberate Atlas
divergence**, more accurate in theory but model was BF16-trained so the
numerical signature differs from vLLM.

## 3. RMSNorm formula and accumulator

**vLLM** uses `GemmaRMSNorm` aliased as `Qwen3NextRMSNorm`
(`qwen3_next.py:38`). `forward_static` (`layernorm.py:322-346`):
```
orig = x.dtype                        # bf16
x = (x + residual) if residual else x # bf16 add, residual = x bf16
x = x.float()                          # → fp32
var = x.pow(2).mean(-1)                # fp32 reduction
x = x * rsqrt(var + eps)               # fp32
x = x * (1.0 + weight.float())         # fp32, weight upcast
x = x.to(orig)                          # → bf16
```
`(1+w)` multiplication is **before** the BF16 downcast.

**Atlas** plain-BF16 path `rms_norm_residual`
(`kernels/gb10/common/rms_norm.cu:177-250`) and FP32 variant
`rms_norm_residual_f32` (`:354-423`): both accumulate `sum_sq` in FP32 via
`warp_reduce_sum` (`:31-36`), compute `rms = rsqrtf(sum_sq/H + eps)`, then
write `pack_bf16x2(xv * rms * (1+wv), …)`. The `(1+w)` multiplication is
also FP32-before-BF16-pack. Algebraically identical to vLLM.

Variance accumulator: **both FP32**. eps applied to mean (vLLM `variance + eps`,
Atlas `sum_sq/H + eps`). HF reference (`modeling_qwen3_next.py:158-166`) matches both.

## 4. eps value

`config.json` has `rms_norm_eps = 1e-6`. vLLM keeps it as Python float (FP64)
all the way into the CUDA op. Atlas casts to f32 at the call site
(`qwen3_ssm/trait_prefill_phase1.rs:31`,
`qwen3_attention/trait_impl/prefill_inner.rs:40`).
`1e-6` is exact in FP32 (within 1 ulp). **No divergence.**

## 5. Explicit BF16→FP32 upcast before reduction

Both engines upcast BF16 → FP32 before the variance reduction. vLLM at
`layernorm.py:339` (`x.float()`); Atlas inside the kernel via
`unpack_bf16x2(...)` returning FP32 floats (`rms_norm.cu:18-21, 64-66`).
Atlas in the FP32-residual mode skips the unpack and reads the FP32 input
directly (`rms_norm.cu:372-375`). **No divergence**: both accumulate in
FP32.

## Flagged bugs / divergences (Atlas-side)

1. **FP32 residual stream is an Atlas-only path.** With
   `ATLAS_FP32_RESIDUAL=1` the embedding is up-cast BF16→FP32 by
   `batched_embed_f32` (`embed_from_argmax.cu:57-70`) and the residual
   buffer carries FP32 across all 48 layers. vLLM keeps the residual in
   BF16. This changes the per-layer residual add precision (FP32 add vs
   BF16 add) — first-token outputs should still match, but **late-layer
   activations differ from vLLM by several BF16 ulps cumulatively**, and
   Qwen3.6 is known to amplify late-layer drift into argmax flips (memory
   index entry `project_qwen36_c1_diagnostic.md`). If reproducing vLLM
   exactly is the goal, **try `ATLAS_FP32_RESIDUAL=0`** to match vLLM's
   BF16 residual semantics.

2. **`rms_norm_residual_f32` writes the FP32 residual from the FP32
   input.** (`rms_norm.cu:414-415`: `res[base] = xv0`.) The first time
   this kernel runs at layer 0 the "input" is the FP32 embedding produced
   by `batched_embed_f32`. The FP32 representation of a BF16 value is
   exact, so this is bit-correct for the lookup. **Not a bug, but worth
   noting**: subsequent layers store the post-add FP32 hidden, which is
   strictly more accurate than vLLM's BF16 stash.

3. **`scale_embeddings` is dispatched but is a no-op for Qwen3.6.**
   `embed_chunk.rs:88` always calls `self.scale_embeddings(...)`. For
   Qwen3.6, `config.embed_scale == 0.0` so
   `scale_embeddings_bf16`/`_fp32` early-returns. This is correct
   behaviour but adds an unconditional dispatch — confirmed match with
   vLLM (which has no scaling at all). **Not a bug.**

4. **All other math matches vLLM exactly** for the embedding + first
   RMSNorm phase. Formula `(1+w)`, FP32 accumulator, FP32 `(1+w)`
   multiply before BF16 cast, eps placement — all identical.

**Bottom line for this phase:** with `ATLAS_FP32_RESIDUAL=0`, the
embedding + first-RMSNorm path is **bit-equivalent** to vLLM to within
BF16 rounding (different reduction order in the warp_reduce vs Inductor
codegen could move a couple of bits). With `ATLAS_FP32_RESIDUAL=1`
(current bench config) Atlas diverges — more accurate per-step but
different numerical fingerprint. The 30%-vs-10/10 cargo_valid gap is
**not** explained by this phase in isolation; look at projection /
attention / MoE kernels next.
