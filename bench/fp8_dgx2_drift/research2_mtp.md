# research2 — MTP K=2 on Qwen3.6-35B-A3B-FP8: cause or amplifier?

**Scope** — Atlas runs Qwen3.6-35B-A3B-FP8 (30 GDN + 10 full-attn hybrid) under MTP K=2 (`--num-drafts 1`). This note audits the MTP code path for (a) draft accept/reject mechanics, (b) SSM state rollback correctness, (c) tool-call / grammar interaction, (d) measured value vs decode-only.

---

## 1. Code path summary

**Two-phase scheduler.** `mtp_step.rs::step_mtp` runs bootstrap-then-verify:
- *Bootstrap* (`mtp_step.rs:33-105`): regular decode → grammar-masked sample → `save_hidden_for_mtp` → `run_mtp_propose_multi` (gated off during `<think>`) → `start_checkpoint_async`.
- *Verify* (`mtp_step.rs:107-148`): `truncate_drafts_at_grammar_boundary` → `step_verify_k{2,3,4}` (or dflash).

Grammar-aware truncation runs **before** verify (`mtp_step.rs:124-132`, ref. arXiv:2512.15834). For K=2 this is a no-op (single draft); for K≥3 it prevents the verifier from accepting a span that crosses `</function>` and silently desyncing the FSM.

**Verify-time pre-sample pipeline (Phase C-2, 2026-05-24).** The load-bearing piece. The verify CUDA graph returns raw GPU argmax per position; pre-Phase C-2 those argmaxes were emitted directly, bypassing mid-word `</think>` defer, grammar bitmask, pin-to-tool-call, forced-token fast-path, `<tool_call>` mask during thinking, etc. `verify_pipeline_helper::verify_pick_all_with_pipeline` now D2H-copies `[K × vocab]` BF16, dequantises each, runs the same 8-stage pipeline as `decode_logits_seq::process_seq_logits`, picks the post-pipeline argmax, and **speculatively advances xgrammar** between positions (rolled back via `gs.rollback(n)` before returning), so position-(i+1)'s bitmask reflects the post-pick-i matcher state. Cost: ~0.8 ms/position × K.

**SSM state rollback (rule-of-three).**
- *MTP K=2/3/4 verify* (`async_chkpt.rs::commit_verify_state_async`): full-accept → scratch → live. Partial-accept → `intermediate[num_accepted-1] → live`. Full-reject → live untouched (pre-verify state preserved by `pre_verify_copy_async`).
- *NGram/self-spec reject* (`async_chkpt.rs::start_rollback_and_checkpoint_async_dispatch`, 80-144): same `intermediate[num_accepted-1]` indexing.
- *Watchdog rollback* (`rollback.rs::rollback_to_boundary`): hybrid models restrict boundary selection to one with a live SSM snapshot via `SsmDecodeRing`, then call `Model::restore_decode_ssm_snapshot`.

All three reuse the SAME `SsmSnapshotPool` GPU D2D machinery (SSOT with Marconi prefix caching). KV rewind is trivial: lowering `seq.seq_len` IS the rewind (attention reads `[0, seq_len)` only).

---

## 2. Is MTP causing or amplifying multi-turn drift on Qwen3.6-35B-A3B-FP8?

### 2.1 Audited and cleared

- **SSM rollback indexing.** Audited 2026-05-23 ([[project_mtp_k2_audit_2026_05_23]]) against vLLM #40880's `EagleProposer` SSM rollback bug. Atlas's K=2 commit matches the canonical pattern; all 30 GDN layers are walked in `async_chkpt.rs:211-257`. NOT a candidate for path corruption.
- **Verify-time logits pipeline.** Phase C-2 closed the biggest gap; live-verified 2026-05-24 vs opencode transcripts. Was a major contributor to "grammar desync, malformed tool calls, mid-word `</think>` cuts".
- **K=3+ single-mask reuse across drafts** (`mtp_head.rs:280-291`). Code warns but uses one mask for every draft; harmless at Qwen3.6 K=2.

### 2.2 Genuinely suspect

**S1 — Tier-1 `</`/whitespace `logit_bias` does NOT apply to MTP verify.** `decode_logits_seq.rs:430-440` pushes `-8.0` biases for token 510 (`</`) and whitespace tokens 220/198/197/256/271 into `logit_bias_local`. These are consumed only by `sample_with_params_history` at line 441. `verify_pick_with_pipeline` runs the pipeline (which applies the grammar bitmask) but returns the post-mask argmax directly — the sampler is never reached. Result: when MTP is active inside `<parameter=KEY>` body with zero chars emitted, the empty-parameter guard is silently disabled. Matches the user-observed `tool_calls[0].function.arguments={}` pattern. **HIGH severity for tool-rich workloads.**

**S2 — Speculative-grammar-advance can desync on forced-token-fastpath termination.** `verify_pipeline_helper.rs:182-191` breaks the speculation loop on `accept_token == false`, but the outer verify step still calls `emit_token` on subsequent positions, which re-calls `accept_token` on a now-terminated matcher → response ends with `gs.accept_token returned false; ending response`. Probability low under tool_choice=auto, higher under tool_choice=required. **MED severity.**

**S3 — `after_verify` trim formula may have regressed from the 2026-05-15 dense-Qwen fix.** [[project_dense_27b_mtp_residual]] documents the original bug — original code trimmed ALL proposed KVs on reject (including the bootstrap entry written by `forward_one` with the accepted token + real `target_hidden`), oscillating `mtp_seq_len` 0↔1 and tanking acceptance to 0.12%. Fix per memory: `num_to_trim = num_drafted.saturating_sub(num_accepted + 1)`. Current `mtp_head.rs:338` reads `num_drafted.saturating_sub(num_accepted)` — i.e. the *unfixed* version. **@human-review**: was the fix reverted, or is the memory describing a different (now-resolved) state? A live `mtp_seq_len` oscillation trace across 100 opencode steps would settle it.

**S4 — Verify D2H + pipeline cost.** ~610 KB D2H per K=2 step on PCIe-attached GB10 LPDDR5X = ~50µs bandwidth + per-position pipeline replay ≈ 1.6ms/step total. Necessary cost of correctness, not a bug.

**S5 — K=3+ mask staleness across drafts.** `mtp_head.rs:280-291` admits one mask is held across all drafts. For Qwen3.6 K=2 (default `num_drafts=1`) this is fine. K≥3 relies on `truncate_drafts_at_grammar_boundary` to catch the boundary-crossing. NOT a Qwen3.6 issue at default config.

### 2.3 Is MTP causal for multi-turn collapse on Qwen3.6?

Validated upstream:
- **27B dense FP8** ([[project_qwen36_27b_degeneration_rootcause]]): MTP-on produced a 30k-tok CSS attractor loop with period ~80 > `CONTENT_LOOP_PERIOD_MAX=64`; MTP-off the same prompt completed cleanly. MTP **amplifies** attractor loops. Leviathan rejection sampling shipped 2026-05-11 ([[project_leviathan_mtp]]) breaks this class — 12% cost vs argmax-MTP, eliminates CSS loops.
- **35B-A3B FP8 MoE**: primary multi-turn mechanism is the MoE expert-routing drift cascade (8/8 → 3/8 at L38) per [[project_qwen36_drift_moe_smoking_gun]], NOT MTP per se. But MTP **uses the model's own hidden states as draft context** — when expert routing flips at deep layers under FP8 drift, the MTP head sees the drifted hidden and proposes accordingly. The proposer is as long-context-degraded as the model, and accept on low-margin tokens *locks them in*. MTP is an amplifier of the underlying drift, not its source.

### 2.4 Is MTP buying material throughput on Qwen3.6-35B-A3B-FP8?

We have **no direct number** for the 35B-A3B variant in current memory. Adjacent points:

| Model | Mode | Throughput | Source |
|---|---|---|---|
| Qwen3.6-27B-FP8 dense | MTP K=2 71% accept | 37.8 tok/s c=1 | [[project_dense_mtp_fix]] |
| Qwen3.6-27B-FP8 dense | no MTP | 25.4 tok/s c=1 | same |
| Qwen3.6-27B-FP8 dense | MTP + Leviathan | 21.3 tok/s | [[project_leviathan_mtp]] (eliminates loops) |
| Nemotron-Super-120B SSM-heavy | MTP K=1 50% accept | 22.3 vs 23.8 baseline (−6%) | [[project_mtp_k1_super120b]] |

For 35B-A3B specifically: A=3B active means decode is already cheap. K=2 verify costs ~2× a decode step (forward + D2H + pipeline). **Break-even is ~50% accept; below 40% MTP is a guaranteed loss AND retains attractor amplification risk.** Live measurement of the per-100-step K2_SUMMARY accept rate during a real opencode transcript is the single experiment that would settle the question.

### 2.5 Upstream state

- **vLLM #39273**: SSM rollback in EagleProposer. OPEN. No clean fix that maps onto Atlas (vLLM's path differs).
- **vLLM #40880**: MTP + GDN + tool calls = malformed args at long context. Atlas K=2 commit pattern is the prescribed fix; issue still open upstream.
- **sgl-project/sglang #18590**: SSM rollback bug. OPEN. SGLang ships per-token SSM checkpointing for the verify span; Atlas effectively has this via `intermediate[]` in `SsmStatePool`.
- **arXiv:2506.01206 ("Mamba Drafters for Speculative Decoding")**: per-token snapshot is canonical. Atlas does this. The unsolved question — what to do when an MTP-driven *multi-step trajectory* is subtly wrong — has no shipped fix. Practical upstream recommendation: drop MTP on hybrid SSM models in favour of NGram or self-spec. Both alternatives exist in Atlas (`step_self_spec`, `step_ngram`).

---

## 3. Ranked top-5 — concrete bugs and recommendations

**1. (HIGH, BUG) — Tier-1 logit_bias bypassed on MTP verify path.** The `-8.0` bias on token 510 (`</`) + whitespace tokens (220/198/197/256/271) that prevents empty `<parameter=KEY></parameter>` bodies is applied only on the non-MTP sampler at `decode_logits_seq.rs:441`. `verify_pick_with_pipeline` runs the pre-sample pipeline (grammar bitmask included) but returns the post-mask argmax directly — sampler bias never applied. *Fix*: thread the Tier-1 biases through `LogitsContext` so `run_pipeline` applies them BEFORE the verify argmax, or apply them in-line in `verify_pick_with_pipeline` when `a.inside_parameter_body && a.param_body_chars_emitted == 0`.

**2. (HIGH, RECOMMENDATION) — Disable MTP for tool-active turns on Qwen3.6-35B-A3B-FP8.** When `require_tool_call=true` or `inside_tool_body=true`, fall back to bootstrap decode (skip propose+verify). Evidence: (a) MTP-driven CSS attractor loops on the 27B sibling, (b) the verify-pipeline gap from #1, (c) MoE routing drift makes draft acceptance low-margin precisely when correctness matters, (d) the forced-token fast-path absorbs most of MTP's gain inside structured tool bodies anyway. Cost: throughput hit on multi-tool turns; benefit: removes attractor amplification at Atlas's most fragile phase.

**3. (MED, BUG) — Verify-pipeline speculative-grammar-advance can desync into mid-call termination.** `verify_pipeline_helper.rs:182-191` breaks the speculation loop on `accept_token=false`, but the outer step still emits later-position tokens, which re-call `accept_token` on a now-terminated matcher → response ends prematurely. *Fix*: when speculative advance fails at position `i`, mark positions `[i+1..K]` INVALID and have the verify step cap acceptance at `i` regardless of verifier argmax. Graceful K-downgrade rather than mid-call termination.

**4. (MED, AUDIT) — Verify `after_verify` trim formula vs the dense-Qwen 2026-05-15 fix.** Memory documents the fix as `num_to_trim = num_drafted.saturating_sub(num_accepted + 1)` (keep bootstrap KV); current `mtp_head.rs:338` reads `.saturating_sub(num_accepted)`. Either (a) memory is stale and current code is intentionally the +0 variant for some reason, or (b) the +1 fix was lost. Settle it by instrumenting `mtp_seq_len` over a 100-step opencode trace and checking for 0↔1 oscillation (signature of the original bug).

**5. (MED, RECOMMENDATION) — Auto-disable MTP under low acceptance.** The atomics in `verify_k2_step.rs:14-32` already track per-100-step accept rate. Add a per-request rolling-window check: if accept rate over the last N=200 verify steps drops below 40%, switch to bootstrap decode for the rest of the request. Rationale: K=2 break-even is ~50%; below 40% MTP is a guaranteed throughput loss AND retains attractor-amplification risk on a model that's *already* drifting. Makes MTP self-correcting under degenerate distributions instead of doubling down.

---

## 4. What we did NOT find

- No evidence Atlas K=2 mis-handles SSM rollback indexing (vLLM #40880 / sglang #18590 patterns audited and cleared in [[project_mtp_k2_audit_2026_05_23]]).
- No evidence MTP propose corrupts grammar state beyond what `truncate_drafts_at_grammar_boundary` catches.
- No evidence MTP K=2 corrupts tool-call XML at the KV/SSM/grammar-state level. The corruption that has been observed (`tool_calls=[]`, mid-word `</think>` cuts, stray `<think>` re-entry) is downstream of MTP, traced to (a) FP8 dequant → MoE routing flips → low-margin tokens, (b) the verify-pipeline gap that Phase C-2 already largely closed, and (c) the residual Tier-1 bias gap flagged in #1 above.

## 5. References

`crates/spark-server/src/scheduler/{mtp_step,verify_k2_step,verify_k3_step,verify_k4_step,verify_pipeline_helper,spec_step,rollback,decode_logits_seq,decode_logits_step,emit_step}.rs`; `crates/spark-model/src/model/trait_impl/{async_chkpt,speculative,verify_c}.rs`; `crates/spark-model/src/layers/mtp_head.rs`. Memories: [[project_mtp_k2_audit_2026_05_23]], [[project_leviathan_mtp]], [[project_dense_27b_mtp_residual]], [[project_qwen36_27b_degeneration_rootcause]], [[project_qwen36_drift_moe_smoking_gun]], [[project_qwen36_fp8_post_think_eos]]. Upstream: arXiv:2506.01206, arXiv:2512.15834; vLLM #39273 #40880; sglang #18590.
