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

/// `CONFIDENCE_RUN_LIMIT` (30) is the historical default; the live limit
/// is `watchdog_params().confidence_run_length` (MODEL.toml-tunable).
pub const CONFIDENCE_RUN_LIMIT: u32 = 30;

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

/// Boundary gate for the forced `</think>` injection. F2 / the
/// thinking-budget cap may *arm* `force_end_thinking` while the model
/// is mid-code-block; injecting `</think>` there would split a
/// statement (the 2026-05-17 thinkbrake bug) and corrupt the answer.
/// Defer the injection until the ``` fence closes — code blocks in
/// reasoning are finite, so the brake then fires cleanly at the block
/// boundary (right after the closing ```), never mid-statement.
/// Outside a fence it fires immediately, exactly as before. In-fence
/// deferral is BOUNDED by `hard_override`: if thinking overran its
/// budget by [`THINK_DEFER_BUDGET_FACTOR`]× (or [`THINK_DEFER_ABS_CEILING`]
/// when no budget is set) while still in a fence, inject anyway — without
/// this a model that writes its whole answer as an in-`<think>` code
/// block keeps `in_code_fence=true` forever and traps the deliverable in
/// reasoning_content (observed 2026-05-17: 3D-chess prompt → 3025
/// reasoning tokens vs 256 budget, 499-char content stub).
pub fn should_inject_think_end(
    force_end_thinking: bool,
    in_code_fence: bool,
    hard_override: bool,
) -> bool {
    force_end_thinking && (!in_code_fence || hard_override)
}

#[cfg(test)]
#[path = "confidence_tests.rs"]
mod confidence_tests;
