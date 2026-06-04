# DFlash γ-Batch Port — Borrowing EAGLE K4 Batched Forward

**Status:** Design / handoff for implementation
**Author:** Friday (architect) — for Claude Code execution
**Date:** 2026-06-04
**Target:** Get DFlash γ-block (K>2) propose+verify actually running, lifting
the current 15.1 tok/s / 50.5% K2-accept baseline toward the 22 tok/s goal.

---

## 0. Current state (verified on disk, do not re-derive)

- **Live tree:** `~/code/atlas`, HEAD `7b5512a` (tag
  `wip/dflash-GOLD-15tok-20260602`). This is the proven baseline:
  **15.1 tok/s, K2 accept 50.5% (96/190)**, no FP8-KV in checkout.
- **Rollback safety:** the GOLD tag is the restore point. No snapshot needed;
  any experimental commits should be tagged before benching.
- **The gap:** `crates/spark-model/src/layers/dflash_head/propose.rs` is a
  **Phase 2.5 STUB** (line ~106). It does NOT run a γ-token batched drafter
  forward — it returns a single draft, so the scheduler
  (`mtp_step.rs:127`, `drafts.len() >= 4` gate) always falls through to
  `step_verify_k2`. That is why we run K2, never the γ path.
- **The verify side is already built:** `verify_dflash_step.rs::step_verify_dflash`
  has correct accept-prefix logic over γ tokens
  (`drafts[i] == verified[i]`, accept prefix, bonus = `verified[first_mismatch]`).
  It is waiting on a propose that actually emits γ drafts.

## 1. Goal

Implement the γ-token batched drafter forward in `propose_drafts()` so it
returns γ (≈16, or a tuned 4–8) draft tokens per step. Route through the
existing `step_verify_dflash` accept-prefix path. Reuse — not reinvent — the
batched GEMV + MoE-batching machinery proven in the EAGLE K4 work.

## 2. What to borrow from EAGLE (source commits)

The batched-forward acceleration that made EAGLE K4 viable lives on commits
**`fa0451d`** (tag `wip/eagle-phase2-ws4-15.4tps-20260603-021953`) and
`4d9a5a1`, NOT on the GOLD tree. Cherry-pick / port these pieces:

| Piece | Source path @ fa0451d | Why we need it |
|---|---|---|
| `w4a16_gemv_batch4` / `_dual_batch4` / `_qg_batch4` | `kernels/gb10/common/w4a16_gemv.cu` | Batched QKV/O GEMV over γ rows in one launch instead of γ serial GEMVs |
| `forward_k4` MoE batching | `crates/spark-model/src/layers/moe/forward_k4.rs` | Batches the γ-row MoE/FFN — the dominant per-verify cost (237ms→124ms on EAGLE) |
| batch wiring | `qwen3_attention/{qkv.rs,attn.rs,init.rs,types.rs}` | How the batched kernels get discovered + dispatched |

**Key parallel:** EAGLE K4 batched 4 verify rows; DFlash γ batches γ drafter
rows through the same shaped kernels. The batching dimension is the only real
difference. Per LTM id155 this port was always the plan (the release-gate note:
"port the batchN GEMV + forward_k4 MoE batching pattern to DFlash's gamma-block
verify path").

## 3. What already exists in the DFlash head (reuse in place)

`crates/spark-model/src/layers/dflash_head/`:
- `forward_block_layer_paged.rs` — per-layer paged-attention body (Option B).
  This is most of Step 3 (a–k) from the propose roadmap already written.
- `precompute_ctx_kv.rs` — `precompute_and_store_context_kv`, the prompt-prefix
  KV population. Reuse for the first-call context write.
- `from_weights.rs` — head struct, FP8 drafter weights, γ, target_layer_ids,
  rope config. All loaded; propose just needs to call into it.

## 4. Implementation roadmap (propose.rs Step 0–5, already commented)

The stub at `propose.rs:106-168` already contains the full step list. Make it
real, borrowing batched kernels from §2:

1. **Step 0/1 — context:** project the 5 target hiddens through `fc` →
   rms_norm → write one ctx token via `reshape_and_cache` (or full prefix via
   `precompute_ctx_kv` on first call).
2. **Step 2 — build γ-token query:** embed token 0 = `last_token`, tokens 1..γ =
   `mask_token_id`, add fc context to row 0 (`combine_hidden_states` semantics;
   verify vs vLLM `qwen3_dflash.py:DFlashQwen3Model.forward`).
3. **Step 3 — γ rows through N drafter layers:** use
   `forward_block_layer_paged` per layer, but swap the serial per-row GEMVs for
   the **batch4-style γ-batched GEMV** ported from EAGLE, and the FFN for the
   **forward_k4-style batched MoE/FFN**. This is where the speedup lives.
4. **Step 4 — final norm + LM head:** `dense_gemm(lm_head_shared)` → `[γ, vocab]`
   → `argmax_bf16` per row → γ token IDs → D2H γ×4 bytes.
5. **Step 5 — state:** `dstate.seq_len += γ + 1`, `last_num_drafted = γ`.

Return `Vec<u32>` of length γ so `mtp_step.rs:127` routes to
`step_verify_dflash`.

## 5. Required kernel handles (resolve via `ctx.gpu.kernel(...)`)

`rms_norm, dense_gemv_bf16, dense_gemm_bf16, rope_qwen3_yarn,
reshape_and_cache_fp8, prefill_attention_paged_fp8_dflash, silu_mul,
residual_add, argmax_bf16, batched_embed` — plus the **ported batched GEMV +
forward_k4 MoE** handles from §2.

## 6. Risks / pitfalls (from LTM)

- **Do NOT enable FP8-KV** (id129): it collapses accept from 50.5% → ~0%. Keep
  the GOLD checkout's no-FP8-KV KV path.
- **Asymmetric q_len (γ) vs k_len (γ + ctx)** — pad q or use a 1-block scratch
  cache for ctx K/V (open question in stub lines 98–100).
- **RoPE position offsets** — ctx K positions map to prior decoded positions;
  q/noise K positions map to `seq_len..+γ` (stub 101–102).
- **Bench last, crib vLLM first** (id151): diff against vLLM
  `qwen3_dflash.py` token-by-token with the nologik harness BEFORE benching.
  Ronald runs all builds/benches — hand off exact commands, do not block.
- **Tag before bench:** commit + tag any γ-attempt so we never lose the GOLD
  baseline again.

## 7. Acceptance / done criteria

1. `propose_drafts()` returns γ drafts; scheduler logs `verify_dflash_step`
   (NOT `verify_k2_step`).
2. Token-for-token parity vs vLLM `qwen3_dflash` reference on the nologik
   harness (≥15/16 γ positions match, per the id139 bar).
3. tok/s ≥ GOLD baseline (15.1) and trending toward 22; accept rate reported.
4. No FP8-KV; GOLD tag intact.
