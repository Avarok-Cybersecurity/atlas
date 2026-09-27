// SPDX-License-Identifier: AGPL-3.0-only

//! The K-position pick loop of `verify_pick_all_with_pipeline`, split out so
//! it can be driven over host logits without a model.

use super::verify_pick_with_pipeline;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::think_commit::{
    SpecThinkTrail, ThinkTokenEnv, advance_thinking_token, spontaneous_think_budget,
};
use crate::scheduler::types::ActiveSeq;

/// Step-start snapshot of the commit state the window advances speculatively
/// (restored on exit — the window only picks; `emit_token` commits).
struct SpecThinkState {
    thinking_tokens: u32,
    in_code_fence: bool,
    force_end_thinking: bool,
    sentence_defer_count: u32,
    consecutive_confident: u32,
    think_watchdog_fires: u32,
    thinking_budget: Option<u32>,
    think_skip_count: u32,
    require_tool_call: bool,
    tool_call_opened: bool,
}

impl SpecThinkState {
    fn capture(a: &ActiveSeq) -> Self {
        Self {
            thinking_tokens: a.thinking_tokens,
            in_code_fence: a.in_code_fence,
            force_end_thinking: a.force_end_thinking,
            sentence_defer_count: a.sentence_defer_count,
            consecutive_confident: a.consecutive_confident,
            think_watchdog_fires: a.think_watchdog_fires,
            thinking_budget: a.thinking_budget,
            think_skip_count: a.think_skip_count,
            require_tool_call: a.require_tool_call,
            tool_call_opened: a.tool_call_opened,
        }
    }

    fn restore(&self, a: &mut ActiveSeq) {
        a.thinking_tokens = self.thinking_tokens;
        a.in_code_fence = self.in_code_fence;
        a.force_end_thinking = self.force_end_thinking;
        a.sentence_defer_count = self.sentence_defer_count;
        a.consecutive_confident = self.consecutive_confident;
        a.think_watchdog_fires = self.think_watchdog_fires;
        a.thinking_budget = self.thinking_budget;
        a.think_skip_count = self.think_skip_count;
        a.require_tool_call = self.require_tool_call;
        a.tool_call_opened = self.tool_call_opened;
    }
}

/// Run the pre-sample pipeline over `k` host-resident logits rows and return
/// the processed pick per position.
///
/// Positions after the first are masked against the matcher state that the
/// EARLIER picks would leave behind (speculative `accept_token`, rolled back
/// at the end so `emit_token` re-advances from a clean matcher).
///
/// Reasoning boundary (2026-09-06): a row may cross `</think>`. The
/// sequence's `inside_thinking` is its state at step START, so without
/// tracking the close every later position was picked free-run and the
/// unconstrained token was then fed to a pristine `required`/schema grammar
/// by `emit_token` — a guaranteed disengage on the first content token. A
/// `</think>` pick now flips the thinking flags for the REST of this loop
/// (mirroring what `emit_token` will do for real), so the first post-think
/// position is picked under the pristine grammar; drafts that disagree are
/// simply rejected by the verifier. The flags are restored on exit — this
/// loop only picks, it never commits.
///
/// Spec-in-think parity (A146, 2026-09-26): the window now advances EVERY
/// piece of commit state a later position's pipeline reads — the picks
/// themselves (pushed onto `output_tokens`: mid-word / sentence-boundary
/// `prev` token, penalty history), `thinking_tokens`, `in_code_fence`,
/// budget / THINK_LOOP arming (SSOT `think_commit::advance_thinking_token`,
/// shared with `process_decode_logits` and `emit_token`), `</think>` resets,
/// spontaneous `<think>`, the post-`</think>` pin inputs — and restores all
/// of it on exit. Because a position's pick already reflects any close that
/// spec-off would force right after an earlier position, an accepted run
/// never needs truncating: a draft that disagrees with the forced token is
/// rejected there. The pipeline's own accumulators are restored too and
/// left per position in `a.spec_think_trail` for `emit_token`.
pub(super) fn pick_positions_from_host(
    buf: &[u8],
    vocab: usize,
    elem_bytes: usize,
    k: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> Vec<u32> {
    let mut picks: Vec<u32> = Vec::with_capacity(k);
    // Snapshot the matcher's history depth BEFORE speculative advances so we
    // roll back exactly the ACTUAL advances afterward. BUG#3 (2026-06-02):
    // stop/EOS and terminated tokens return true from `accept_token` WITHOUT
    // advancing the matcher, so a count of `accept_token`→true calls would
    // over-rewind. `emit_token` (run after this helper) re-advances from the
    // restored, clean state.
    let grammar_steps_before = a.grammar_state.as_ref().map(|gs| gs.num_history_steps());
    let think_flags_before = (a.inside_thinking, a.think_ended, a.think_just_ended);
    // A144: tool-body state per position. The `<tool_call>` opener bias is
    // stripped INSIDE a tool body (`strip_in_tool_opener_bias`, via
    // `penalty_params_for`), so a window that opens or closes a call must
    // re-evaluate it at every later position against the tokens picked
    // earlier in the SAME window — exactly what `emit_token` →
    // `update_tool_param_state` will do on the accept path. Restored on exit.
    let tool_body_before = a.inside_tool_body;
    // Spec-in-think parity (A146, AVAROK_DFLASH_SPEC_THINK): every piece of
    // per-token commit state the pipeline READS at a later position must be
    // advanced here per position, exactly as the commit twins
    // (`process_decode_logits` / `emit_token`) will advance it, and restored
    // on exit. The pipeline reads: `thinking_tokens` (F2 >=400 gate, A4
    // floor, forced-`</think>` hard override), `in_code_fence` (fence
    // deferral), `force_end_thinking` (budget / THINK_LOOP arming),
    // `output_tokens` (mid-word mask + sentence-boundary gate via `.last()`,
    // penalty history), `think_just_ended` / `require_tool_call` /
    // `tool_call_opened` (post-think `<tool_call>` pin), and the
    // spontaneous-`<think>` budget.
    let think_state_before = SpecThinkState::capture(a);
    let out_len_before = a.output_tokens.len();
    let think_env = ThinkTokenEnv {
        code_fence_token: ctx.code_fence_token,
        think_loop_enabled: !ctx.sampling.disable_watchdogs
            && ctx.watchdog.enable_think_loop_watchdog,
        watchdog: ctx.watchdog,
    };
    a.spec_think_trail.clear();

    for i in 0..k {
        let slice = &buf[i * vocab * elem_bytes..(i + 1) * vocab * elem_bytes];
        // The window's earlier picks are pushed onto `output_tokens` below
        // (speculative history), so every `output_tokens.len() + verify_pos`
        // consumer (seed offset, min_tokens mask, forced-token, A144 base
        // bias) already sees this position's emitted length: pass 0.
        let pick = verify_pick_with_pipeline(slice, false, vocab, a, ctx, 0);
        picks.push(pick);
        // What the pipeline left in its accumulators for THIS position —
        // re-applied by `emit_token` if (and only if) this position commits.
        a.spec_think_trail
            .push_back(SpecThinkTrail::capture(a, pick));

        // ── Mirror of the commit transitions (emit_token / decode) ──
        // Spontaneous `<think>`: enters thinking, never pushed.
        if !a.inside_thinking && a.think_start_token == Some(pick) {
            a.inside_thinking = true;
            a.think_ended = false;
            a.think_skip_count = 0;
            a.thinking_budget = Some(spontaneous_think_budget(a));
            continue;
        }
        // Stray `</think>` outside thinking: skipped, never pushed.
        if !a.inside_thinking && ctx.think_end_token == Some(pick) {
            continue;
        }
        if a.require_tool_call && a.tool_call_start_token == Some(pick) && !a.inside_thinking {
            a.require_tool_call = false;
            a.tool_call_opened = true;
        }
        // Twin of the 512-token safety clear (decode / emit_token).
        if a.require_tool_call && a.output_tokens.len() > 512 {
            a.require_tool_call = false;
        }
        let is_eos = a.eos_tokens.contains(&pick);

        // `</think>` closes the span for every later position. It is never
        // fed to the matcher (the grammar only sees content tokens), so no
        // speculative advance here either. Same resets as the commit twins.
        if a.inside_thinking && ctx.think_end_token == Some(pick) {
            a.inside_thinking = false;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            a.consecutive_confident = 0;
            a.in_code_fence = false;
            a.think_ended = true;
            a.think_just_ended = true;
            a.output_tokens.push(pick);
            continue;
        }
        if a.inside_thinking {
            // SSOT `advance_thinking_token` (thinking_tokens, fence, budget
            // arm, THINK_LOOP) — decode runs it for a (discarded) EOS too.
            let history_len = a.output_tokens.len();
            advance_thinking_token(a, pick, history_len, think_env, false);
        } else {
            // `emit_token` clears the post-`</think>` one-shot on the first
            // content token; without this a later position would re-pin.
            a.think_just_ended = false;
            // Mirror `update_tool_param_state`'s opener/closer transitions (a
            // no-op inside thinking, like the real one).
            if a.tool_call_start_token == Some(pick) {
                a.inside_tool_body = true;
            } else if a.tool_call_end_token == Some(pick) {
                a.inside_tool_body = false;
            }
        }
        // Decode never pushes an EOS it keeps generating past (an EOS it
        // honours ends the sequence, so later positions are moot).
        if !is_eos {
            a.output_tokens.push(pick);
        }

        // Speculatively advance the matcher with `pick[i]` so the next
        // position's bitmask reflects post-emit state. Skip on the last
        // position (no next position to mask) and when the seq has no
        // grammar (nothing to advance).
        if i + 1 < k
            && let Some(ref mut gs) = a.grammar_state
            && !a.inside_thinking
        {
            // Matcher advance can fail if `pick` is not in the current
            // bitmask. If our pipeline correctly applied the bitmask,
            // pick is the argmax over masked logits → MUST be in the
            // bitmask → advance MUST succeed. The defensive check
            // exists for forced-token fast-path returns where the
            // grammar may have terminated; those legitimately can't
            // advance further.
            if !gs.accept_token(pick) {
                tracing::debug!(
                    pick,
                    i,
                    "verify_pick: grammar speculative advance refused — pipeline picked a token outside the current bitmask. \
                     This indicates a stale bitmask in the pipeline or a forced-token fastpath that terminated grammar. \
                     Stopping speculation here; the real `accept_token` in emit_token will fail and end the response."
                );
                break;
            }
            // accept_token advanced the matcher as a side effect; the rollback
            // below counts the ACTUAL advances from matcher history (BUG#3).
        }
    }

    // Roll back exactly the ACTUAL speculative advances (history delta) so the
    // matcher returns to its pre-call state; `emit_token` then re-advances it
    // normally. BUG#3: counting from accept_token→true calls over-rewinds when
    // a stop/EOS/terminated token (which returns true WITHOUT advancing) lands
    // in the verified span.
    if let (Some(before), Some(gs)) = (grammar_steps_before, a.grammar_state.as_mut()) {
        let advanced = gs.num_history_steps().saturating_sub(before);
        if advanced > 0 {
            gs.rollback(advanced);
        }
    }
    // Same discipline for the thinking flags: `emit_token` owns the real
    // `</think>` transition on the accept path.
    (a.inside_thinking, a.think_ended, a.think_just_ended) = think_flags_before;
    a.inside_tool_body = tool_body_before;
    think_state_before.restore(a);
    a.output_tokens.truncate(out_len_before);

    picks
}
