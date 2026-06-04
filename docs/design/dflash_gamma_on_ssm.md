# DFlash γ-Block — Implementation Plan on Merged SSM Tree

**Status:** Implementation handoff for Claude Code
**Supersedes:** the EAGLE-borrow premise in `dflash_eagle_batch_port.md`
  (DISPROVEN — see §2). Pairs with `dflash_eagle_batch_port_phase0.md` findings.
**Date:** 2026-06-04
**Branch:** `dflash-on-ssm` @ `0c9831c` (GOLD baseline merged onto nologik
  `spec_ssm`). Compiles clean, baseline holds: **15.2 tok/s, K2 accept 50.5%**.

---

## 0. The one-line goal

Implement the γ-token batched drafter forward in
`crates/spark-model/src/layers/dflash_head/propose.rs::propose_drafts` so it
returns γ drafts, routing through the existing
`scheduler/mtp_step.rs:127` (`drafts.len() >= 4 → step_verify_dflash`) gamma
path — and make the SSM/GDN target state roll back correctly on partial
draft acceptance. Target: break past the K2 ceiling toward 22 tok/s.

## 1. Why this is now viable (what the merge bought us)

The target Qwen3.6-27B is **3:1 GDN:full-attention** (confirmed: linear layers
carry `linear_conv_kernel_dim=4`, split K/V head counts = gated-delta-net, NOT
plain linear attention; the `layer_types` label "linear_attention" hides this).
γ>1 speculation on a GDN target REQUIRES recurrent-state rollback on draft
rejection — that is the blocker that sank every prior attempt.

**nologik's `spec_ssm` branch (now merged) provides exactly that.** Verified on
the tree at `crates/spark-model/src/layers/ops/`:
- `ssm_gdn_a.rs:322` `gdn_decode_chunk2` — "Saves intermediate H_1 state for
  rollback on draft rejection."
- `ssm_gdn_b.rs:19` `gdn_decode_chunk3` — "Saves 2 intermediate H states
  (H_1, H_2) for rollback on draft rejection."
- `ssm_gdn_b.rs:289` `conv1d_update_chunk2` — "Saves intermediate conv_state
  (after token 0) for rollback."
- WY-variant decode kernels `gdn_decode_wy2/wy3/wy4/wy17` for batched chunks.

These are the rollback primitives the gamma verify path needs. We do NOT have
to write SSM rollback from scratch — wire to these.

## 2. What is DEAD (do not pursue — Phase 0 already ruled out)

- **EAGLE `w4a16_gemv_batch4` / `_dual_batch4` / `_qg_batch4`:** NVFP4-weight
  kernels; DFlash drafter is BF16/FP8 dense. Cannot borrow. `dense_gemm(M=γ)`
  already batches all γ rows in one call.
- **`forward_k4` MoE batching:** drafter has a DENSE FFN, not MoE. Irrelevant.
- **DFlash per-layer body (Step 3a–3k):** ALREADY COMPLETE in
  `dflash_head/forward_block_layer_paged.rs`. Do not re-port.

## 3. Open questions Phase 0 already RESOLVED (do not re-open)

- **Asymmetric q_len (γ) vs k_len (γ+ctx):** handled by the existing paged-attn
  path. No design work.
- **RoPE offsets:** Phase I v2 already stamps ctx positions correctly.

## 4. The actual work (ordered)

1. **Fill the propose stub** (`propose.rs`, currently Phase 2.5b scaffold →
   empty-Vec fallthrough). Implement Steps 0–5 from the in-file roadmap:
   project target hiddens → fc → build γ-token query (token0=last_token,
   1..γ=mask_token_id) → run γ rows through the drafter layers via the
   EXISTING `forward_block_layer_paged` → final norm + lm_head → argmax per
   row → return `Vec<u32>` of length γ.
2. **Wire GDN rollback into the verify seam.** `step_verify_dflash` accepts a
   prefix and rejects the rest. On the TARGET model the GDN recurrent + conv
   state advanced by γ during verify must roll back to the accepted-prefix
   length. Use `gdn_decode_chunk2/chunk3` (saves H states) +
   `conv1d_update_chunk2` (saves conv state) to snapshot before the speculative
   chunk and restore to `num_accepted`. This is the make-or-break correctness
   step — wrong rollback = garbage state = accept collapse.
3. **Verify drafter YaRN config** (one real risk from Phase 0): confirm
   `dflash_head/from_weights.rs` builds `yarn_inv_freq` from the DRAFTER's own
   config (factor=64, orig_max=4096), NOT the target model's. Wrong yarn →
   accept ~0%.

## 5. Method (Ronald's standing rules — LTM id151, id155)

- **Crib vLLM FIRST, bench LAST.** Diff against vLLM `qwen3_dflash.py`
  (`DFlashQwen3Model.forward`, `combine_hidden_states`) token-by-token with the
  nologik token-diff harness before any bench.
- **Friday/Claude write code; Ronald runs ALL builds/benches.** Hand off exact
  commands; never run cargo build/bench.
- **Tag before bench.** Current safety net: GOLD tag
  `wip/dflash-GOLD-15tok-20260602` + `dflash-ssm-merge-*`. Tag any γ attempt
  before benching so the 15.2 baseline is never lost.

## 6. Acceptance criteria

1. `propose_drafts` returns γ drafts; scheduler logs `verify_dflash_step`, not
   `verify_k2_step`.
2. GDN/conv state rolls back correctly on partial accept — coherent output, no
   degeneration.
3. Token parity vs vLLM `qwen3_dflash` on nologik harness (≥15/16 γ positions).
4. tok/s ≥ 15.2 baseline and climbing toward 22; report accept rate. No FP8-KV
   (id129: collapses accept). GOLD tag intact.

## 7. First step for Claude (read-only, before coding)

Read `ssm_gdn_a.rs` + `ssm_gdn_b.rs` rollback kernel signatures and
`verify_dflash_step.rs`, then confirm the exact snapshot/restore call sequence
the verify seam needs. Report that wiring plan, THEN implement propose §4.1.
