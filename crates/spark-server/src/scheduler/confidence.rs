// SPDX-License-Identifier: AGPL-3.0-only

//! F2 confidence-run + code-fence pure helpers, extracted from
//! `helpers.rs` to keep that file ≤500 LoC. These drive the F2
//! confidence early-stop and the safe-boundary `</think>` injection
//! gate; they are pure parity/accumulator functions, unit-tested
//! directly without any `ActiveSeq` / logits mocking.

use super::helpers::watchdog_params;

/// Flip `in_fence` when the just-sampled token `tok` is the model's
/// atomic ``` code-fence token. `fence_tok == None` (tokenizer has no
/// single fence token) disables the guard: the fence state can never
/// become `true`, so F2 keeps its prior behaviour (fail-open, PCND —
/// no implicit default, the absence is explicit and inert).
///
/// Pure parity function — the single source of truth for fence
/// tracking, called from the decode token-accept path.
pub fn toggle_code_fence(in_fence: bool, tok: u32, fence_tok: Option<u32>) -> bool {
    match fence_tok {
        Some(f) if f == tok => !in_fence,
        _ => in_fence,
    }
}

/// `CONFIDENCE_RUN_LIMIT` is the default streak length before F2's
/// confidence-early-stop arms `</think>`. The live limit is
/// `watchdog_params().confidence_run_length` (MODEL.toml-tunable).
///
/// 2026-05-23 sweep: 30 → 60. With the project-wide `max_thinking_budget`
/// bump to 2048 reasoning chains genuinely span hundreds of tokens, and
/// the previous 30-token streak fired inside legitimate confident
/// reasoning (e.g. dictating boilerplate path strings, listing imports,
/// or a model-card cite). 60 still catches genuine collapse — a stuck
/// model emits ≥60 high-confidence tokens within ~2 s — without firing
/// during normal extended thinking.
pub const CONFIDENCE_RUN_LIMIT: u32 = 60;

/// F2 confidence-run accumulator. Given whether the current token is
/// high-confidence (top-1 softmax ≥ 0.95) and the prior consecutive
/// run length, return `(new_run, should_arm_force_end)`.
///
/// Pure accumulator — runs the SAME inside and outside a ``` fence.
/// We deliberately keep *detecting* inside code: a model that drafts
/// an unbounded code block in its reasoning still needs braking. What
/// must NOT happen is the forced `</think>` landing mid-statement —
/// that boundary decision is [`should_inject_think_end`] below, which
/// defers the injection until the fence closes (a safe boundary).
pub fn confidence_run_step(confident: bool, prev_run: u32) -> (u32, bool) {
    if confident {
        let run = prev_run + 1;
        (run, run >= watchdog_params().confidence_run_length)
    } else {
        (0, false)
    }
}

/// In-fence deferral budget factor — see [`should_inject_think_end`].
/// 3× budget tolerates a legit in-think code block; beyond that a hard
/// cut beats dumping the whole answer.
pub const THINK_DEFER_BUDGET_FACTOR: u32 = 3;
/// Absolute in-fence deferral ceiling when no thinking budget is set
/// (F2/THINK_LOOP armed force_end with `thinking_budget=None`).
pub const THINK_DEFER_ABS_CEILING: u32 = 2048;
/// Maximum decode steps to defer the forced `</think>` injection
/// after `force_end_thinking` is armed, while waiting for a sentence-
/// boundary token (`.`, `!`, `?`, `\n`). Past this many tokens with no
/// boundary in sight, inject anyway — the model is probably emitting
/// digit columns / identifier lists / unpunctuated prose, and the
/// budget-overrun penalty already dominates aesthetics.
///
/// 2026-05-23 sweep: opencode-session.md showed all 28 `<think>`
/// blocks ending mid-sentence (e.g. `"create thep"`, `"a ping/pong en"`).
/// CGR (arXiv 2509.07820) prescribes deferring the close to the
/// nearest natural boundary; 64 tokens is roughly 2-4 sentences of
/// runway, which lines up with the observed half-sentence overshoot.
pub const MAX_SENTENCE_DEFER_TOKENS: u32 = 64;

/// Boundary gate for the forced `</think>` injection. F2 / the
/// thinking-budget cap may *arm* `force_end_thinking` while the model
/// is mid-statement; injecting `</think>` there would split a sentence
/// (the 2026-05-23 opencode "create thep" / "ping/pong en" bug) and
/// corrupt the reasoning trail. This gate has three deferrals layered
/// from cheapest to most aggressive:
///
/// 1. **In-code-fence defer** (existing): never split a ``` block —
///    code is finite, brake fires after the closing ```.
/// 2. **Sentence-boundary defer** (2026-05-23): outside a fence, wait
///    until the previously-emitted token is a sentence boundary
///    (`.`/`!`/`?`/`\n`, per [`crate::scheduler::helpers::boundary_token_mask`]).
/// 3. **`hard_override`** caller-computed escape hatch: when thinking
///    overran the budget by [`THINK_DEFER_BUDGET_FACTOR`]× / hit
///    [`THINK_DEFER_ABS_CEILING`] / has been deferring for
///    [`MAX_SENTENCE_DEFER_TOKENS`] consecutive steps without finding
///    a boundary, inject anyway. Without (3) a model that writes its
///    whole answer as an in-`<think>` code block keeps
///    `in_code_fence=true` forever and traps the deliverable in
///    reasoning_content (observed 2026-05-17: 3D-chess prompt → 3025
///    reasoning tokens vs 256 budget, 499-char content stub).
pub fn should_inject_think_end(
    force_end_thinking: bool,
    in_code_fence: bool,
    at_sentence_boundary: bool,
    hard_override: bool,
) -> bool {
    if !force_end_thinking {
        return false;
    }
    if hard_override {
        return true;
    }
    if in_code_fence {
        return false;
    }
    at_sentence_boundary
}

#[cfg(test)]
#[path = "confidence_tests.rs"]
mod confidence_tests;
