// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `confidence.rs` (F2 confidence-run + code-fence
//! pure helpers). Split out of `helpers_tests.rs` when the F2 helpers
//! moved to `confidence.rs` to keep both files ≤500 LoC. Logical child
//! of `confidence` via `#[path]`; `use super::*` resolves to
//! `confidence.rs` items.

use super::*;

// ── Code-fence guard for the F2 confidence early-stop ──────────────
// Regression coverage for the 2026-05-17 thinkbrake bug: the model
// drafting a ```python block inside <think> produced 30+ consecutive
// ≥0.95 tokens, tripping F2 and force-injecting </think> mid-line.

const FENCE: u32 = 71093; // Qwen3.x atomic ``` token

#[test]
fn fence_toggles_on_fence_token() {
    assert!(
        toggle_code_fence(false, FENCE, Some(FENCE)),
        "``` opens fence"
    );
    assert!(
        !toggle_code_fence(true, FENCE, Some(FENCE)),
        "``` closes fence"
    );
}

#[test]
fn fence_unchanged_by_non_fence_token() {
    assert!(!toggle_code_fence(false, 42, Some(FENCE)));
    assert!(toggle_code_fence(true, 42, Some(FENCE)));
}

#[test]
fn fence_guard_disabled_when_no_fence_token() {
    // Tokenizer split ``` → guard inert, never enters a fence.
    assert!(!toggle_code_fence(false, FENCE, None));
}

#[test]
fn f2_arms_after_confidence_run_limit_tokens() {
    // Pure accumulator: CONFIDENCE_RUN_LIMIT consecutive confident
    // tokens arm the brake. Constant raised from 30 → 60 in the
    // 2026-05-23 sweep — the assertion tracks the constant, so this
    // test remains stable across future tuning.
    let mut run = 0;
    let mut fired = false;
    for _ in 0..CONFIDENCE_RUN_LIMIT {
        let (next, fire) = confidence_run_step(true, run);
        run = next;
        fired |= fire;
    }
    assert_eq!(run, CONFIDENCE_RUN_LIMIT);
    assert!(
        fired,
        "CONFIDENCE_RUN_LIMIT consecutive confident tokens must arm F2"
    );
}

#[test]
fn f2_run_breaks_on_non_confident_token() {
    let (run, fire) = confidence_run_step(false, 25);
    assert_eq!(run, 0);
    assert!(!fire);
}

#[test]
fn f2_accumulates_inside_code_too() {
    // Detection runs everywhere — code is finite and must still be
    // brakeable. (Mid-statement safety is the *injection* gate's job,
    // see `defer_*` tests below — NOT suppression of detection.)
    // Pre-CONFIDENCE_RUN_LIMIT step: run advances, fire stays false.
    let (run, fire) = confidence_run_step(true, CONFIDENCE_RUN_LIMIT - 2);
    assert_eq!(run, CONFIDENCE_RUN_LIMIT - 1);
    assert!(!fire);
    // At-limit step: run hits the cap and fire flips true.
    let (run, fire) = confidence_run_step(true, CONFIDENCE_RUN_LIMIT - 1);
    assert_eq!(run, CONFIDENCE_RUN_LIMIT);
    assert!(
        fire,
        "F2 arms even inside a fence; injection is what defers"
    );
}

// ── should_inject_think_end: the safe-boundary defer gate ─────────
// This is the core of the 2026-05-17 in-fence fix + 2026-05-23
// sentence-boundary fix: the forced </think> may be armed mid-stream,
// but it must not be *injected* mid-code-fence (would split a
// statement) nor mid-sentence (would split a thought).
//
// Signature: should_inject_think_end(
//     force_end_thinking,
//     in_code_fence,
//     at_sentence_boundary,
//     hard_override,
// )

#[test]
fn defer_injection_while_in_code_fence() {
    // Inside fence, armed, AT boundary, no override → still defer
    // (fence wins over boundary because splitting code is worse).
    assert!(
        !should_inject_think_end(true, true, true, false),
        "armed brake must NOT inject </think> mid-code-fence (would split a statement)"
    );
}

#[test]
fn inject_once_fence_closes_at_sentence_boundary() {
    // Outside fence, armed, at boundary, no override → fire.
    assert!(
        should_inject_think_end(true, false, true, false),
        "armed brake fires cleanly once the ``` fence has closed AND a sentence boundary is reached"
    );
}

#[test]
fn defer_outside_fence_when_not_at_sentence_boundary() {
    // 2026-05-23 sweep: previously the brake would fire immediately
    // outside a fence (3-arg signature, in_fence=false → inject).
    // Now we ALSO require `at_sentence_boundary` — without it the
    // brake defers, letting the model finish its current thought.
    assert!(
        !should_inject_think_end(true, false, false, false),
        "armed brake must NOT inject </think> mid-sentence (would corrupt reasoning)"
    );
}

#[test]
fn hard_override_breaks_unbounded_in_fence_defer() {
    // The 2026-05-17 chess regression: model writes its whole
    // answer as a ```block inside <think>, fence never closes,
    // budget brake deferred forever. hard_override must force the
    // injection even mid-fence.
    assert!(
        should_inject_think_end(true, true, false, true),
        "armed + in-fence + budget massively overrun must HARD-inject </think>"
    );
    // Not armed → still nothing, even with override.
    assert!(!should_inject_think_end(false, true, false, true));
}

#[test]
fn hard_override_breaks_unbounded_sentence_defer() {
    // 2026-05-23 sweep: when sentence_defer_count reaches
    // MAX_SENTENCE_DEFER_TOKENS the caller folds it into
    // hard_override. Without this, a model emitting digits /
    // identifiers without periods would defer forever.
    assert!(
        should_inject_think_end(true, false, false, true),
        "armed + outside fence + no boundary + hard_override → force-inject"
    );
}

#[test]
fn no_injection_when_not_armed() {
    // Not armed: every (in_fence, boundary, override) permutation
    // must keep the gate closed.
    for &in_fence in &[false, true] {
        for &at_boundary in &[false, true] {
            for &hard_override in &[false, true] {
                assert!(
                    !should_inject_think_end(false, in_fence, at_boundary, hard_override),
                    "not-armed must not inject: in_fence={in_fence}, at_boundary={at_boundary}, hard_override={hard_override}"
                );
            }
        }
    }
}

#[test]
fn boundary_at_least_one_path_eventually_fires() {
    // Smoke test: from each (in_fence, at_boundary) starting point,
    // there exists SOME (force=true, hard_override) input that fires.
    // Ensures the gate is not pathologically stuck for any state.
    assert!(should_inject_think_end(true, false, true, false)); // boundary path
    assert!(should_inject_think_end(true, false, false, true)); // override path
    assert!(should_inject_think_end(true, true, false, true)); // fence + override
    assert!(should_inject_think_end(true, true, true, true)); // all true → fire
}
