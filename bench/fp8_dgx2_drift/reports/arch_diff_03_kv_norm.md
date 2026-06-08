# Arch Diff #03 — K/V/Q-norm (Qwen3-Next, BF16 path)

Function-by-function comparison of the sub-layer RMSNorm that sits **between QKV projection and RoPE** in Qwen3-Next-80B-A3B (HF: `qwen3_next`). Atlas BF16 prefill paged path vs vLLM 0.x main.

---

## 1. Where the norm is called

### vLLM (`vllm/model_executor/models/qwen3_next.py:775-805`)
```python
self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # alias for GemmaRMSNorm
self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
# ...
q = self.q_norm(q.view(-1, self.num_heads,    self.head_dim)).view(-1, self.num_heads*self.head_dim)
k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads*self.head_dim)
q, k = self.rotary_emb(positions, q, k)
```
Note: `Qwen3NextRMSNorm` is imported as `GemmaRMSNorm` (line 38) — i.e. the **offset-from-1** variant.

### Atlas (`crates/spark-model/src/layers/qwen3_attention/prefill/paged.rs:198-234`)
- Q path: deinterleave + Q-norm fused in `deinterleave_qg_split_qnorm` (line 132) when `gated`; else `ops::rms_norm` over `q_contiguous` in-place (line 198, 211).
- K path: `ops::rms_norm` in-place over `k_contiguous` (line 224).
- V path: only ran if `self.v_norm_weight.is_some()` — Gemma-4 only (line 245-257).
- Then RoPE consumes the same `k_contiguous` / `q_contiguous` BF16 buffers (line 314+).

---

## 2. Formula — `(1 + w)` vs `w`

Both engines agree on **`(1 + w)`** for Qwen3-Next.

- vLLM `GemmaRMSNorm.forward_static` (`layernorm.py:344`): `x = x * (1.0 + weight.float())`.
- Atlas `rms_norm.cu:106`: `pack_bf16x2(xv0 * rms * (1.0f + wv0), …)`. Confirmed for **all** non-`_abs` variants in the file; Gemma-4 uses the `_abs` siblings that drop the +1 (rms_norm.cu:581, 588, 658).

Atlas's `(1+w)` matches vLLM. **No bug here.**

---

## 3. FP32 vs BF16 between RMSNorm and RoPE — the key question

### vLLM
`GemmaRMSNorm.forward_static` (`layernorm.py:339-346`):
```python
x = x.float()                                # promote to FP32
variance = x.pow(2).mean(...)
x = x * torch.rsqrt(variance + eps)
x = x * (1.0 + weight.float())
x = x.to(orig_dtype)                         # ← ROUND BACK TO BF16
return x
```
RoPE then receives BF16 tensors (`rotary_embedding/base.py:156-163` — `ops.rotary_embedding` is an in-place call on the original BF16 `query`/`key` buffers).

**vLLM rounds to BF16 between RMSNorm and RoPE.**

### Atlas
`rms_norm.cu:45-113` — input BF16, FP32 accumulator for `sum_sq`, FP32 `rms = rsqrtf(...)`, multiplies in FP32, then **`pack_bf16x2` at line 106** rounds to BF16 on write. The buffer is the same `k_contiguous` slot that RoPE reads next.

**Atlas rounds to BF16 between RMSNorm and RoPE — identical to vLLM.**

### Verdict for audit
The previous audit comment that "Atlas rounds to BF16 between RMSNorm and RoPE — potential bug" is **CONFIRMED MATCHES vLLM**. Not a divergence, not a bug source. Both engines truncate at the same point. RoPE is BF16-in, BF16-out on both sides.

---

## 4. V-norm — Qwen3-Next has none

- vLLM (`qwen3_next.py:775-803`): no `v_norm` member, V flows straight from `qkv.split` into `self.attn(q, k, v)`.
- Atlas (`paged.rs:245`): `if let Some(v_norm_w) = self.v_norm_weight.as_ref()` — Gemma-4 only. For Qwen3-Next the weight is `None` and the branch is skipped.

**Both engines agree: no v_norm on Qwen3-Next.**

---

## 5. Q-norm timing relative to RoPE

- vLLM: `q_norm` applied per-head (view as `[-1, num_heads, head_dim]`) **before** RoPE (`qwen3_next.py:798 → 805`).
- Atlas (gated): fused `deinterleave_qg_split_qnorm` (`paged.rs:132`) deinterleaves Q from QG and normalizes per-head in one kernel, before RoPE at line 314+.
- Atlas (non-gated): plain `ops::rms_norm` over `(nq*n, hd)` rows (line 198-208) — same shape contract.

Both engines: Q-norm → RoPE, per-head granularity, BF16 between. **Agrees.**

There is a MiniMax-specific `q_norm_full` branch in Atlas (line 182-196) that does a *single* RMS over the entire `[nq*hd]` per token instead of per-head. That branch is not taken for Qwen3-Next (the loader sets `q_norm_full = None` and populates the per-head `q_norm` weight); not a Qwen drift source.

---

## 6. Numerical knobs

| Field                         | vLLM                                  | Atlas                                       | Match? |
|------------------------------|---------------------------------------|---------------------------------------------|--------|
| `eps`                        | `config.rms_norm_eps` = `1e-6`        | `ctx.config.rms_norm_eps as f32` (paged.rs:40) | ✅ |
| Accumulator dtype            | FP32 (`x.float()`)                    | FP32 (`sum_sq`, `rms`, products)            | ✅ |
| `mean(x²)`                   | `x.pow(2).mean(dim=-1)`               | `sum_sq / hidden_size` (rms_norm.cu:96)     | ✅ |
| `rsqrt`                      | `torch.rsqrt(var+eps)`                | `rsqrtf(mean+eps)` (rms_norm.cu:96)         | ✅ |
| Vector width                 | tensor-op (cuDNN/PyTorch fused)       | 2-wide BF16 packed loads (rms_norm.cu:65,104) | n/a |
| Output dtype                 | BF16 (`.to(orig_dtype)`)              | BF16 (`pack_bf16x2`)                        | ✅ |
| Reduction tree               | torch.mean (deterministic per-row)    | warp-shuffle XOR + cross-warp via shmem    | order differs |
| In-place semantics           | functional return                     | input==output pointer (in-place buffer)     | n/a |

**Subtle divergence:** Atlas's warp-shuffle `__shfl_xor_sync` reduction (rms_norm.cu:32-36) produces FP32 sums in a different add-order than PyTorch's row-wise mean. Non-associative FP32 → ≤1 ULP difference per token, well below BF16 round-off floor (~3.9e-3). Not a drift source.

---

## Findings

1. **No bug in the (1+w) formula** — both engines apply offset-from-1, matching HF `Qwen3NextRMSNorm`.
2. **No FP32-vs-BF16 inter-op divergence** — both engines round to BF16 between RMSNorm and RoPE. The audit's standing question is now resolved: **Atlas matches vLLM here**, not a leading drift candidate.
3. **No v_norm on Qwen3-Next on either side.** Atlas correctly gates on `v_norm_weight.is_some()` (Gemma-4 only).
4. **Q-norm timing is identical** (post-deinterleave, pre-RoPE, per-head granularity).
5. **`eps` matches** at `1e-6`.
6. **Accumulator precision matches** (FP32 throughout intermediate compute).
7. **Reduction order differs** but bounded by FP32 ULP, below BF16 quantization noise — not a drift cause.

**Recommendation:** RMSNorm-K/Q is **not** the FP8 drift root cause. Look upstream (QKV projection FP8 dequant, see arch_diff_02 if it exists) or downstream (RoPE `cos/sin` precision, attention softmax — see project_qwen36_phase2b_softmax_expf memory which flagged late-layer FP8 KV magnitude noise). Close this audit branch.

---

**Files cited**
- `/home/nologik/vllm/vllm/vllm/model_executor/models/qwen3_next.py:38,775-805`
- `/home/nologik/vllm/vllm/vllm/model_executor/layers/layernorm.py:305-346` (GemmaRMSNorm)
- `/home/nologik/vllm/vllm/vllm/model_executor/layers/rotary_embedding/base.py:133-164` (forward_cuda)
- `/workspace/atlas-mtp/crates/spark-model/src/layers/qwen3_attention/prefill/paged.rs:130-257,314+`
- `/workspace/atlas-mtp/kernels/gb10/common/rms_norm.cu:45-113` (rms_norm), `:581,588,658` (Gemma-4 abs variants)
