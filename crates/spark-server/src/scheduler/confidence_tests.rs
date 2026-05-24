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
fn f2_arms_after_30_confident_tokens() {
    // Pure accumulator: 30 consecutive confident tokens arm the brake.
    let mut run = 0;
    let mut fired = false;
    for _ in 0..CONFIDENCE_RUN_LIMIT {
        let (next, fire) = confidence_run_step(true, run);
        run = next;
        fired |= fire;
    }
    assert_eq!(run, CONFIDENCE_RUN_LIMIT);
    assert!(fired, "30 consecutive confident tokens must arm F2");
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
    let (run, fire) = confidence_run_step(true, 29);
    assert_eq!(run, 30);
    assert!(
        fire,
        "F2 arms even inside a fence; injection is what defers"
    );
}

// ── should_inject_think_end: the safe-boundary defer gate ─────────
// This is the core of the 2026-05-17 fix: the forced </think> may be
// armed mid-code-block, but it must not be *injected* there.

#[test]
fn defer_injection_while_in_code_fence() {
    assert!(
        !should_inject_think_end(true, true, false),
        "armed brake must NOT inject </think> mid-code-fence (would split a statement)"
    );
}

#[test]
fn inject_once_fence_closes() {
    assert!(
        should_inject_think_end(true, false, false),
        "armed brake fires cleanly once the ``` fence has closed"
    );
}

#[test]
fn hard_override_breaks_unbounded_in_fence_defer() {
    // The 2026-05-17 chess regression: model writes its whole
    // answer as a ```block inside <think>, fence never closes,
    // budget brake deferred forever. hard_override must force the
    // injection even mid-fence.
    assert!(
        should_inject_think_end(true, true, true),
        "armed + in-fence + budget massively overrun must HARD-inject </think>"
    );
    // Not armed → still nothing, even with override.
    assert!(!should_inject_think_end(false, true, true));
}

#[test]
fn no_injection_when_not_armed() {
    assert!(!should_inject_think_end(false, false, false));
    assert!(!should_inject_think_end(false, true, false));
}
