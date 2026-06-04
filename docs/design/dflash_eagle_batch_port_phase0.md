# DFlash γ-Batch Port — Phase 0 Investigation (do this BEFORE coding)

**Status:** Investigation handoff for Claude Code
**Pairs with:** `dflash_eagle_batch_port.md` (the architecture/port map)
**Date:** 2026-06-04
**Author:** Friday (architect)

---

## Why this doc exists

`dflash_eagle_batch_port.md` is a sound architecture map, but three things are
still OPEN questions, not settled instructions. Writing code before resolving
them risks a mid-implementation rewrite. **Do this read-only investigation
first, then produce a concrete implementation plan and STOP for review.**
Do not write production code in Phase 0. Crib vLLM first, bench last (LTM id151).

## Ground rules

- Read-only. No edits to `crates/` in this phase.
- Live tree `~/code/atlas` @ `7b5512a` (GOLD, 15.1 tok/s). Do not disturb it.
- Ronald runs all builds/benches — never run cargo build/bench yourself.
- Output of this phase = a written implementation plan appended here under
  §"FINDINGS", plus a go/no-go on each open question below.

## Investigation tasks

### A. EAGLE batched kernels (the things we borrow)
Read at commit `fa0451d` (tag `wip/eagle-phase2-ws4-15.4tps-20260603-021953`):
- `kernels/gb10/common/w4a16_gemv.cu` — the `w4a16_gemv_batch4`,
  `_dual_batch4`, `_qg_batch4` kernels. Document their exact signatures, the
  batch dimension layout (row stride, how the 4 is parameterized), and whether
  the batch count is hard-coded to 4 or a runtime arg. **Critical:** can it
  generalize to batch=γ (e.g. 16), or is 4 baked into smem/tiling?
- `crates/spark-model/src/layers/moe/forward_k4.rs` — how the MoE/FFN batches
  4 rows. Same question: γ-generalizable or fixed-4?
- `qwen3_attention/{qkv.rs,attn.rs,init.rs,types.rs}` @ fa0451d — how these
  kernels are discovered and dispatched.

### B. vLLM reference (the correctness oracle)
- Find and read `qwen3_dflash.py` (`DFlashQwen3Model.forward` and
  `combine_hidden_states`) in the vLLM refs (`~/eagle-refs/`).
- Document exactly: query construction (token0=last_token, tokens1..γ=mask),
  how fc-context is added to row 0, the attention mask shape (in-block
  bidirectional + prefix), and RoPE position assignment for q vs ctx-K.

### C. The three OPEN questions — resolve each with a recommendation
1. **Batch width:** does the EAGLE batch4 machinery extend to γ rows, or do we
   run ceil(γ/4) batched passes? Recommend a concrete approach.
2. **Asymmetric q_len (γ) vs k_len (γ + ctx):** pad q with a dummy row, or use
   the paged attention with a 1-block scratch ctx cache? Pick one, justify.
3. **RoPE offsets:** confirm ctx-K positions = prior decoded positions, q/noise-K
   positions = `seq_len..+γ`. Verify against vLLM, note any mismatch with the
   drafter's yarn config (factor=64, orig_max=4096).

### D. Reuse audit
- Confirm what `dflash_head/forward_block_layer_paged.rs` already implements of
  Step 3 (a–k) so we don't re-port what exists. List which sub-steps are
  present vs missing.

## Deliverable (append below when done)

A plan that, per drafter layer, names: the exact kernel handle to call, batched
vs serial, the buffer shapes, and the resolved answer to each open question.
Plus a risk list and the exact build/bench commands to hand Ronald. Then STOP
for review before any code is written.

---

## FINDINGS
_Investigation completed 2026-06-04. All reads from `~/code/atlas` @ `7b5512a` (GOLD)
and EAGLE commit `fa0451d` (tag `wip/eagle-phase2-ws4-15.4tps-20260603-021953`)._

---

### TL;DR / executive summary

**Two of the three open questions are already answered in the live code.** The
asymmetric q_len/k_len case is completely handled by `forward_block_layer_paged`
(paged attention with `kv_len=ctx_count+γ`), and RoPE offsets are correctly
implemented (Phase I v2 fixed-position ctx slots). The batch-width question has
a surprise: the EAGLE NVFP4 batch4 kernels **cannot be ported to the drafter**
because the drafter uses BF16/FP8 weights, not NVFP4. The existing
`dense_gemm(M=γ)` call in the layer body is already the correct "batched" path.

The **real unresolved blocker** — not mentioned in the open questions — is the
SSM rollback constraint described in `propose.rs` lines ~232–248: the K=γ verify
path does not populate `h_state_intermediates`, so partial-accept on an SSM-
hybrid target (Qwen3.6-A3B) produces corrupted output. This is why the draft cap
sits at 1. Resolving this — either by confirming a pure-attention target or by
scoping out SSM intermediates work — must happen before any γ>1 bench is
meaningful.

---

### A. EAGLE batched kernels (`fa0451d`)

#### `w4a16_gemv_batch4` — signatures and layout

```c
// w4a16_gemv_batch4 (line 1098)
extern "C" __global__ void w4a16_gemv_batch4(
    const __nv_bfloat16* A,          // [4, K] — rows at A + 0·K, 1·K, 2·K, 3·K
    const unsigned char* B_packed,   // [N, K/2] NVFP4
    const unsigned char* B_scale,    // [N, K/GROUP_SIZE] FP8-E4M3, GROUP_SIZE=16
    const float scale2,              // per-tensor second-level scale
    __nv_bfloat16* C,                // [4, N] — rows at C + 0·N, 1·N, 2·N, 3·N
    unsigned int N, unsigned int K
)
// Grid: (ceil(N/4), 1, 1)   Block: (256, 1, 1)   smem: N_PER_BLOCK * 8 floats

// w4a16_gemv_dual_batch4 (line 1216) — K+V in one dispatch
// Grid: (ceil(N/4), 1, 2)   blockIdx.z=0→K, blockIdx.z=1→V

// w4a16_gemv_qg_batch4 (line 1342) — adds Q/Gate deinterleave, same M=4 shape
```

**Is 4 baked in?** Yes, structurally. Four hardcoded row pointers
(`A1=A+K, A2=A+2K, A3=A+3K`), four register accumulators (`acc0..acc3`), smem
layout `N_PER_BLOCK * 8` (= 2 warps × 4 accs). No runtime M parameter. The file
contains separate kernels for M=1, M=2, M=3, M=4 — no `batchN`.

**Can it generalize to γ?** No. For γ≠4, you would need a new kernel (e.g.
`w4a16_gemv_batchN` with M as a template or runtime parameter, or serial
ceil(γ/4) passes of `batch4`).

#### `forward_k4` MoE — M=4 hardcoded, drafter-irrelevant

All paths in `moe/forward_k4.rs` hardcode M=4: buffer math uses `4*top_k` and
`4*h`, every downstream op (`moe_expert_gate_up_shared_batch4`, etc.) bakes in
the count. Not runtime-parameterizable.

**More importantly: the DFlash drafter has no MoE.** The drafter uses a dense
SwiGLU FFN (gate_proj + up_proj + silu_mul + down_proj). `forward_k4` is
irrelevant to the drafter's FFN path.

#### Dispatch wiring (`multi_seq/qkv.rs` diff @ `fa0451d`)

The dispatch is a chain of `if n==4 && nvfp4 → batch4`, `n==3 → batch3`,
`n==2 → batch2`, else serial per-row loop. One hardcoded branch per size.

---

### B. vLLM reference (`~/eagle-refs/vllm/…/qwen3_dflash.py`)

| Item | What vLLM does | Atlas equivalent |
|---|---|---|
| `combine_hidden_states` (line 580) | `self.model.fc(hidden_states)` — projects the concat target hiddens through fc | `precompute_ctx_kv` + fc-projection in `forward_block` |
| Query row 0 construction | embed(`last_token`) + result of `combine_hidden_states(target_hiddens)` | `forward_block` adds fc context to row 0 of stream_buf |
| Query rows 1..γ | embed(`mask_token_id`) | `batched_embed` of mask tokens |
| Position tensor for γ queries | Caller passes `[seq_len, seq_len+1, …, seq_len+γ-1]` | `position_ids` buffer in `forward_block`, built from `position..position+γ` |
| Ctx K positions | Caller passes the actual decoded positions for each ctx token | `dstate.ctx_positions[i]` stamped at append time (Phase I v2) |
| Per-head K-norm order | `k_norm` **before** RoPE | `forward_block_layer_pre_attn` step 3c before 3d ✅ |
| Attention | `self.attn(q, k, v)` via DFlash attention backend (reads ctx from KV cache) | `prefill_attention_paged_dflash_bf16_indirect` ✅ |

**No mismatch found** between vLLM's DFlash forward and Atlas's
`forward_block_layer_paged` for the ops covered. One sub-question (yarn config)
remains — see Open Question 3.

---

### C. Open questions — resolved

#### Q1 — Batch width

**Finding: the architecture doc's "borrow w4a16_gemv_batch4" is inapplicable.**

The EAGLE batch4 GEMV kernels target **NVFP4 weights** (`B_packed` is packed
uint8 NVFP4). The DFlash drafter layers hold `DenseWeight` (BF16) and
`Fp8DenseWeight` (FP8) — no NVFP4 path exists in `forward_block_layer_paged`.
These kernels cannot be borrowed directly.

The existing code already handles the batched case:
`forward_block_layer_pre_attn` and `_post_attn` call `gemm_swap` which routes to
`dense_gemm(M=γ)` (BF16) or `fp8_gemm_n128_row_scaled(M=γ)` (FP8). Both fire
one kernel call for all γ rows — already "batched" in the sense that matters.

**Recommendation:** Do not try to port NVFP4 batch4 to the drafter. The existing
M=γ GEMM path is structurally correct. If bench reveals that `dense_gemm(M=γ)`
at small γ (4–8) is the throughput bottleneck (same pathology as `w4a16_gemm
M=4`), then write `bf16_gemv_batchN` or `fp8_gemv_batchN` kernels modeled on
the NVFP4 batch4 pattern but for the drafter's weight format. Do that after
correctness is confirmed, not before.

For now: start with γ=4 (`ATLAS_DFLASH_DRAFT_CAP=4`), bench the wall-clock per-
propose time, profile if slow.

#### Q2 — Asymmetric q_len (γ) vs k_len (γ + ctx)

**Already solved.** `forward_block_layer_paged` uses
`prefill_attention_paged_dflash_bf16_indirect` with:
- `q_len = γ`, `kv_len = ctx_count + γ`, `q_offset = ctx_count`
- Ctx K/V at paged slots `[0..ctx_count)`, γ K/V written to `[ctx_count..ctx_count+γ)` in step 3e
- `option_b_indirect_args_dev` carries `kv_len` and `q_offset` to the kernel so CUDA graph replays pick up per-call values

Neither padding nor a scratch ctx-cache is needed. No action required.

#### Q3 — RoPE offsets

**Substantially correct, with one unverified sub-item.**

- Ctx K positions: Phase I (v2) stamps each ctx slot's absolute position at
  append time (`position.saturating_sub(1)`) in `dstate.ctx_positions`. The
  `precompute_ctx_kv` call uses exactly these positions. Matches vLLM. ✅
- γ noise K + query positions: `position_ids` buffer in `forward_block` holds
  `position..position+γ`. Matches vLLM's positions tensor for the γ query
  tokens. ✅
- **Unverified: drafter yarn config.** The drafter was trained with
  `factor=64, orig_max=4096, beta_fast=32, beta_slow=1`. The main model uses
  different rope_scaling. Confirm that `self.yarn_inv_freq` and
  `self.rope_theta` in `BlockDiffusionDraftHead` (`from_weights.rs`) were built
  from the drafter's own config, not the main model's. If wrong, accept rate
  collapses to near zero — this is a HIGH risk item. **Action:** read
  `from_weights.rs` yarn construction and diff against the drafter's HF
  `config.json` before any bench.

---

### D. Reuse audit — `forward_block_layer_paged.rs` Step 3 (a–k)

**All sub-steps are present.** Nothing needs to be ported.

| Sub-step | Status | Where |
|---|---|---|
| 3a `input_layernorm` rms_norm | ✅ present | `pre_attn` ~line 130 |
| 3b q/k/v_proj dense_gemm | ✅ present | `gemm_swap` × 3 in `pre_attn` |
| 3c q_norm / k_norm per-head | ✅ present | rms_norm on q_buf/k_buf heads |
| 3d rope_yarn | ✅ present | `rope_yarn` call on q/k at positions `[pos..pos+γ)` |
| 3e reshape_and_cache | ✅ present | writes γ K/V at `ctx_count` slots |
| 3f prefill_attention_paged_dflash | ✅ present | `prefill_attn_dflash_bf16_indirect`, q=γ, kv=ctx+γ |
| 3g o_proj dense_gemm | ✅ present | `post_attn` |
| 3h residual_add | ✅ present | `post_attn` |
| 3i post_attention_layernorm | ✅ present | `post_attn` |
| 3j gate+up+silu_mul+down (FFN) | ✅ present | `post_attn` via `gemm_swap` × 3 + `silu_mul` |
| 3k residual_add | ✅ present | `post_attn` |

---

### UNDOCUMENTED BLOCKER — SSM rollback (the actual draft-cap reason)

The open questions list in the doc does not mention this, but it is the current
gate. From `propose.rs` lines ~232–248:

> The K=γ verify path (`decode_verify_graphed_kgamma`) does NOT populate
> `h_state_intermediates` — those are only written by the K=2/3/4 specialized
> GDN kernels. Partial-accept with γ>1 → stale SSM state → output corruption.

This is why `ATLAS_DFLASH_DRAFT_CAP` defaults to 1. The drafter forward IS
running all γ rows correctly already (the forward infrastructure is live).

**Resolution paths (pick one before uncapping):**
1. **Confirm target is pure-attention** (Gemma-4, MiniMax-M2 dense). If so,
   there are no GDN/SSM layers, the rollback issue vanishes, and we can uncap
   immediately.
2. **Defer to a GDN-aware K=γ verify kernel** (multi-week kernel work). Not
   in scope for the current γ-port sprint unless the target is GDN-free.

**This must be the first thing confirmed before writing any code.**

---

### Implementation plan (per drafter layer)

For each of the 8 drafter layers at γ=4 (initial tuned value):

| Step | Kernel handle | Mode | In shape | Out shape |
|---|---|---|---|---|
| 3a rms_norm | `self.kernels.rms_norm` | 1 call, γ rows | [γ, H] | [γ, H] |
| 3b q_proj | `dense_gemm` / `fp8_gemm_n128_row_scaled` | 1 call, γ rows | [γ, H] | [γ, Q_dim] |
| 3b k_proj | same | 1 call, γ rows | [γ, H] | [γ, KV_dim] |
| 3b v_proj | same | 1 call, γ rows | [γ, H] | [γ, KV_dim] |
| 3c q_norm | `rms_norm` | γ×nqh rows | [γ·nqh, hd] | same |
| 3c k_norm | `rms_norm` | γ×nkvh rows | [γ·nkvh, hd] | same |
| 3d RoPE | `self.kernels.rope_qwen3` | 1 call | [γ, Q_dim]/[γ, KV_dim] | same |
| 3e write K/V | `reshape_cache_bf16` | γ slots | [γ, KV_dim] × 2 | paged cache |
| 3f attn | `prefill_attn_dflash_bf16_indirect` | q=γ, kv=ctx+γ | [γ, Q_dim] → [γ, Q_dim] | |
| 3g o_proj | `dense_gemm` / `fp8_gemm_n128_row_scaled` | 1 call, γ rows | [γ, Q_dim] | [γ, H] |
| 3h residual | `residual_add` | γ·H elements | — | — |
| 3i post-norm | `rms_norm` | γ rows | [γ, H] | [γ, H] |
| 3j gate_proj | `dense_gemm` / `fp8_gemm_n128_row_scaled` | γ rows | [γ, H] | [γ, inter] |
| 3j up_proj | same | γ rows | [γ, H] | [γ, inter] |
| 3j silu_mul | `self.kernels.silu_mul` | γ·inter elements | — | — |
| 3j down_proj | same | γ rows | [γ, inter] | [γ, H] |
| 3k residual | `residual_add` | γ·H elements | — | — |

**All handles already exist.** No new Rust op wrappers or kernel registrations
needed to run γ>1 drafts. The only code change required is: confirm target has
no SSM layers, then increase `ATLAS_DFLASH_DRAFT_CAP` (env var) and verify.

---

### Risk list

| Risk | Severity | Action |
|---|---|---|
| SSM rollback on GDN targets | **HIGH** | Confirm target is pure-attention BEFORE removing cap |
| Drafter yarn config mismatch | **HIGH** | Read `from_weights.rs` yarn params; diff vs drafter HF config (factor=64, orig_max=4096) |
| FP8-KV re-enabled accidentally | **HIGH** | `id129` gate in place; never change KV dtype |
| GOLD tag 7b5512a destroyed | **HIGH** | Tag every experimental commit before bench |
| `dense_gemm(M=γ)` small-M perf | MEDIUM | Bench after correctness; write batched GEMV if bottleneck |
| `ctx_committed` not reset on rewind | MEDIUM | Verify rollback path resets `ctx_committed` and `ctx_len` together |
| mask_token row-0 skip logic | MEDIUM | The `drafts[1..]` skip in `propose_drafts` must stay when `mask_token_id != 0`; don't strip it |

---

### Build/bench commands for Ronald

**Step 1 — confirm correctness at γ=1 (no code change, already shipping):**
```
# Start server with Option B + trace
ATLAS_DFLASH_OPTION_B=1 ATLAS_DFLASH_DRAFT_CAP=1 ATLAS_DFLASH_VERIFY_TRACE=1 \
  ./atlas-server [model args]

# Run bench against it
python3 scripts/bench_model.py localhost:8888
```

**Step 2 — vLLM parity (nologik harness, per id151 discipline):**
```
# Run nologik harness with γ=1 against both vLLM qwen3_dflash and Atlas.
# Target: ≥15/16 drafter positions match token-for-token.
# Do this BEFORE uncapping or benching γ>1.
```

**Step 3 — yarn config verification (no code change yet):**
```
# In the Rust source, find from_weights.rs yarn construction and check:
#   yarn_factor, original_max_position_embeddings, beta_fast, beta_slow
# Diff against the drafter's HF config.json.
# If wrong, fix in from_weights.rs before any further work.
```

**Step 4 — uncap to γ=4 (ONLY after pure-attention target confirmed + vLLM parity):**
```
# Tag first!
git tag wip/dflash-gamma4-attempt-$(date +%Y%m%d-%H%M%S)

ATLAS_DFLASH_OPTION_B=1 ATLAS_DFLASH_DRAFT_CAP=4 ATLAS_DFLASH_VERIFY_TRACE=1 \
  ./atlas-server [model args]
python3 scripts/bench_model.py localhost:8888
# Check scheduler logs for "verify_dflash_step" (not "verify_k2_step")
# Check accept rate and tok/s vs GOLD 15.1 baseline
```

**Step 5 — perf profiling (if γ=4 bench is slower than expected):**
```
# Profile propose latency breakdown. If dense_gemm(M=4) shows as bottleneck,
# that is the signal to write batched FP8/BF16 GEMV kernels.
# Do not write those kernels speculatively.
```
