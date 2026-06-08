# Agent D — MTP K2 verify investigation

## TL;DR

**True K2 acceptance rate is ~41.5%, not "near zero" — the log is misleading**: REJECTs are logged on every step but ACCEPTs only when `seq.seq_len.is_multiple_of(50)` (`verify_k2_step.rs:119`). Computing from `seq_len` deltas (8043 → 14062 = 6019 tokens advanced across 2489 rejects), accepts ≈ (6019 − 2489)/2 ≈ **1765 ACCEPTs vs 2489 REJECTs (≈ 41.5%)** — normal for Qwen-style MTP-1. That said, two real defects exist and likely degrade opencode flow even though correctness is nominally preserved:

1. **H1 + H3 confirmed**: the verify path uses **raw GPU argmax of unprocessed logits**. The MTP draft proposes through `mtp_grammar_mask_for(...)` (grammar bitmask applied — `mtp_head/forward.rs:382-463`) — but the **target** in `decode_verify_graphed` returns raw argmax IDs (`verify_k2_step.rs:36-51`, see TODO at L49-50). No mid_word/post_close/tool_during_think/forced_token/F2_confidence/pin_tool_call logit processors run, and no temperature/top_p/top_k sampling — those are wired into `process_seq_logits` only on the non-MTP decode path (`decode_logits_step.rs:78-97`).
2. **H6 partial**: after REJECT, `emit_token(v0)` does push to `output_tokens`, run grammar `accept_token`, content-loop watchdog, `<|im_start|>` hard-stop (`emit_step.rs:104-203`). So those late-stage gates DO fire on the verified token. BUT pre-emit logit shaping is entirely bypassed.

Together: in opencode flows with `enable_thinking=on` + grammar, the target's `</think>` mid-word mask, post-close `<think>` mask, and require_tool_call pin are **silently disabled on every spec-decode token** (both accepted and rejected — accept emits the draft, reject emits the raw target argmax). This easily explains session.md degeneration: stray `<think>` re-entry, garbled tool-call JSON like `test-rust-ax{"v19/Cargo.toml"}`, mid-word emissions.

## Per-hypothesis verdict

- **H1 — MTP draft missing grammar bitmask: PARTIAL.** The MTP head DOES apply `grammar_bitmask` CPU-side (`mtp_head/forward.rs:382-463`) via `mtp_grammar_mask_for(a)` wired through `verify_k2_step.rs:103-117/152-167`. BUT `mtp_grammar_mask_for` returns `None` when `a.inside_thinking` is true (`spec_step.rs:338-350`), so during `<think>` blocks the draft is unconstrained — combined with H3 this desyncs.
- **H2 — distribution divergence from Phase-2c: NOT REFUTED but unlikely dominant.** Both target and MTP head consume the same patched paths (FP8 `__expf` softmax, RNE BF16 dequant). 41% accept rate is normal.
- **H3 — target verify skips logit processors: CONFIRMED (main bug).** `decode_verify_graphed` returns GPU argmax — no `f2_confidence`/`mid_word`/`post_close`/`tool_during_think`/`forced_think_end`/`pin_tool_call`/`forced_token`/`grammar_bitmask` stage runs. Pipeline lives in `process_seq_logits` (`decode_logits_seq.rs`), invoked only from `decode_logits_step.rs:82`.
- **H4 — drafts biasing target: REFUTED.** Drafts sent only via `ep_broadcast_cmd` and echoed in `tokens_k2`; verify just runs the model on those tokens and reads argmax. No logit_bias injection.
- **H5 — rollback skips side effects: NEGATIVE.** Reject path does `seq_len -= 1; tokens.pop(); commit_verify_state_async(seq, 1, 2); save_hidden_for_mtp(0,0); trim_proposer_state(0,0)` (`verify_k2_step.rs:127-148`). Grammar matcher was never advanced on the draft (`accept_token` only fires in `emit_token`), so no rollback needed. Correct.
- **H6 — rollback skips post-decode pipeline: CONFIRMED.** On reject, `emit_token(v0)` runs watchdog + grammar advance but **pre-sample logit pipeline never runs** (same root cause as H3 viewed from reject side).

## Token-pattern evidence

`prev_draft=90` proposed 1867× consecutively, all rejected. Also `4754` (252×), `1152` (74×), `35480` (32×). Classic stuck-MTP signature: the draft head sees a hidden state generated from a token stream the target wouldn't have produced under its post-sample constraints, so the two distributions diverge.

## Recommended next steps (prioritized)

1. **Run the logit pipeline on K2 verify output before the argmax compare** (highest impact). Either D2H copy verify logits and run `process_seq_logits` over `v0`, OR re-use `sample_token_with_grammar` on verify-position-0 logits. Cost ≈ 0.8 ms/step (per TODO at `verify_k2_step.rs:49-51`). Without this, MTP breaks the contract that every emitted token has been gated by mid_word/post_close/forced_token/pin_tool_call/grammar.
2. **Fast mitigation while #1 lands**: serve Qwen3.6 opencode flows with `--num-drafts 0` (disable MTP). Or gate MTP off when `tools.len() > 0` OR `enable_thinking=true`. Throughput drops to baseline, but every token now flows through `process_seq_logits` and degeneration goes away.
3. **Sample the ACCEPT log every step** (5 min, prevent future misdiagnosis). Replace `is_multiple_of(50)` with a periodic-summary `info!`: "K2 last 100 steps: accepts=X rejects=Y rate=Z%".
4. **Re-fill bitmask between draft positions when num_drafts > 1** (deferred — K=2 path unaffected; `mtp_head.rs:285-290` already warns).

Key files: `/workspace/atlas-mtp/crates/spark-server/src/scheduler/verify_k2_step.rs`, `/workspace/atlas-mtp/crates/spark-server/src/scheduler/mtp_step.rs`, `/workspace/atlas-mtp/crates/spark-server/src/scheduler/spec_step.rs:338-404`, `/workspace/atlas-mtp/crates/spark-server/src/scheduler/decode_logits_step.rs:78-97`, `/workspace/atlas-mtp/crates/spark-server/src/scheduler/logit_processors/mod.rs`, `/workspace/atlas-mtp/crates/spark-server/src/scheduler/emit_step.rs:104-203`, `/workspace/atlas-mtp/crates/spark-model/src/layers/mtp_head/forward.rs:382-463`.
