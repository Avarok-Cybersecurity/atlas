// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for scheduler::helpers (loop/fence/F2 detectors).
//! Split out of helpers.rs to keep it ≤500 LoC (CI file-size-cap).
//! Logical child of `helpers` via `#[path]`; `use super::*` resolves
//! to helpers.rs items exactly as before the split.

use super::*;

#[test]
fn detects_period_8_triple_repeat() {
    let pat: Vec<u32> = (1..=8).collect();
    let mut tokens: Vec<u32> = (0..40).collect();
    tokens.extend(pat.iter()); // r1
    tokens.extend(pat.iter()); // r2
    tokens.extend(pat.iter()); // r3
    assert!(detect_thinking_token_loop(&tokens));
}

#[test]
fn rejects_two_repeats() {
    // Even with >= MIN_TOKENS tokens total, only two copies of a
    // period-5 block must not trigger (noise + double is not a
    // degenerate loop).
    let pat: Vec<u32> = (100..=104).collect();
    let mut tokens: Vec<u32> = (0u32..50).collect();
    tokens.extend(pat.iter()); // r1
    tokens.extend(pat.iter()); // r2 only
    assert!(!detect_thinking_token_loop(&tokens));
}

#[test]
fn rejects_numbered_list_reasoning() {
    // Legitimate thinking content: 80 distinct tokens, no repeat.
    let tokens: Vec<u32> = (0u32..80).collect();
    assert!(!detect_thinking_token_loop(&tokens));
}

#[test]
fn detects_short_period_fence_loop() {
    // Simulates `Running ``` bash cd X && cargo test ``` ` as a
    // 10-token repeat. Need at least THINK_LOOP_MIN_TOKENS=48
    // total tokens for the detector to even evaluate, so pad
    // with unique prefix tokens first.
    let pat: Vec<u32> = vec![7, 6, 5, 4, 3, 2, 1, 0, 9, 8];
    let mut tokens: Vec<u32> = (100u32..150).collect(); // prefix pad
    for _ in 0..4 {
        tokens.extend(pat.iter());
    }
    assert!(detect_thinking_token_loop(&tokens));
}

#[test]
fn detects_fence_body_with_varying_prefixes() {
    // The real attractor: fence body (tokens 100..110) is stable
    // but connective prefixes (Running vs Executing) differ
    // between iterations. A strict contiguous-period detector
    // misses this; the substring-repeat detector must catch it.
    let fence: Vec<u32> = vec![100, 101, 102, 103, 104, 105, 106, 107, 108, 109];
    let prefixes: [&[u32]; 4] = [
        &[200, 201],      // "Running:"
        &[202, 203],      // "Executing:"
        &[204, 205, 206], // "I need to run:"
        &[207],           // "Run:"
    ];
    let mut tokens: Vec<u32> = (0..30).collect();
    for pre in prefixes.iter() {
        tokens.extend(pre.iter());
        tokens.extend(fence.iter());
    }
    assert!(
        detect_thinking_token_loop(&tokens),
        "stable fence body across varying prefixes must be detected"
    );
}

// ── Content-phase loop detector tests (Claude Code 2026-04-26 fix) ──

#[test]
fn content_loop_detects_sentence_triple_repeat() {
    // Simulates "I see I've been creating Cargo.toml files but the
    // user hasn't given me a task. Let me wait for their
    // instructions." as a 22-token sentence repeating 3× — exactly
    // the Claude Code 2026-04-26 degeneration. Must fire.
    let sentence: Vec<u32> = (1000..1022).collect();
    let mut tokens: Vec<u32> = (0..100).collect(); // prior content
    tokens.extend(sentence.iter()); // r1
    tokens.extend(sentence.iter()); // r2
    tokens.extend(sentence.iter()); // r3
    assert!(
        detect_content_token_loop(&tokens),
        "22-token sentence repeating 3× must trigger content-loop watchdog"
    );
}

#[test]
fn content_loop_rejects_short_responses() {
    // Below CONTENT_LOOP_MIN_TOKENS — must not fire even on a
    // visible repeat. The watchdog should give short responses
    // breathing room.
    let pat: Vec<u32> = (1..=10).collect();
    let mut tokens: Vec<u32> = (50..80).collect();
    tokens.extend(pat.iter());
    tokens.extend(pat.iter());
    tokens.extend(pat.iter());
    assert!(
        !detect_content_token_loop(&tokens),
        "responses under {} tokens must not trigger watchdog",
        CONTENT_LOOP_MIN_TOKENS
    );
}

#[test]
fn content_loop_rejects_legitimate_prose() {
    // 200 distinct tokens of prose — no repeat. Must not fire.
    let tokens: Vec<u32> = (0u32..200).collect();
    assert!(
        !detect_content_token_loop(&tokens),
        "legitimate prose with no repeat must not trigger watchdog"
    );
}

#[test]
fn content_loop_rejects_two_repeats() {
    // Two copies of a 30-token block with prior context — common
    // in legitimate "the user said X. The user said X again."
    // exposition. Should NOT fire (need 3+ repeats).
    let sentence: Vec<u32> = (500..530).collect();
    let mut tokens: Vec<u32> = (0..100).collect();
    tokens.extend(sentence.iter());
    tokens.extend(sentence.iter()); // r2 only
    assert!(
        !detect_content_token_loop(&tokens),
        "two repeats in content must not trigger (need 3)"
    );
}

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

// ── Digit-normalized content-loop watchdog ───────────────────────
// Regression for the 2026-05-17 Qwen3.6-27B greedy degeneration:
// `- B(46) = N\n- B(47) = M\n …` — fixed line template, varying
// integer payload, runs to max_tokens. Convention: structural token
// ids 1..=11, numeric ids 100..=199; mask len 1100 (prefix noise
// ids 900..=990 are out of the numeric range → structural).

fn numeric_mask() -> Vec<bool> {
    let mut m = vec![false; 1100];
    for (i, slot) in m.iter_mut().enumerate() {
        *slot = (100..=199).contains(&i);
    }
    m
}

/// 12-token template `[1..=6, <num>, 7..=11]`; `num` varies each
/// repeat so the exact detector cannot match, but normalization
/// collapses every repeat to an identical period.
fn varying_template_stream(repeats: u32) -> Vec<u32> {
    let mut t: Vec<u32> = (900u32..990).collect(); // 90 structural-noise prefix
    for k in 0..repeats {
        t.extend([1, 2, 3, 4, 5, 6]);
        t.push(100 + k); // distinct numeric payload per repeat
        t.extend([7, 8, 9, 10, 11]);
    }
    t
}

#[test]
fn norm_fires_on_varying_numeric_template() {
    let t = varying_template_stream(5);
    let mask = numeric_mask();
    assert!(
        !detect_content_token_loop(&t),
        "exact detector must miss: integer tokens differ every repeat"
    );
    assert!(
        detect_content_token_loop_normalized(&t, &mask),
        "normalized detector must catch the fixed template (5 repeats >= 4)"
    );
}

#[test]
fn norm_rejects_3item_list_and_pure_columns() {
    let mask = numeric_mask();

    // (a) Only 3 repeats — below CONTENT_LOOP_NORM_MIN_REPEATS (4):
    // a legitimate 3-item numbered list must not hard-stop.
    let three = varying_template_stream(3);
    assert!(
        !detect_content_token_loop_normalized(&three, &mask),
        "3 repeats < NORM_MIN_REPEATS=4 must not fire"
    );

    // (b) Pure-number column (period has no structural token):
    // structural prefix keeps global has_struct true, so the
    // per-period needle requirement is what must reject it.
    let mut col: Vec<u32> = (900u32..990).collect();
    for k in 0..6 {
        col.extend([100 + k; 12]); // 12 numeric tokens, no structural
    }
    assert!(
        !detect_content_token_loop_normalized(&col, &mask),
        "pure-number period (no structural token) is the exact path's job"
    );

    // (c) Pure-prose period (no numeric token): early-out on
    // !has_sentinel — left to the exact detector.
    let mut prose: Vec<u32> = (900u32..960).collect();
    for _ in 0..6 {
        prose.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }
    assert!(
        !detect_content_token_loop_normalized(&prose, &mask),
        "pure-prose period (no numeric) must defer to exact detector"
    );
}

#[test]
fn exact_prose_loop_still_caught_regression() {
    // Byte-identical period x4, no mask: the EXACT detector must
    // still fire — guards that detect_token_loop_with_period's
    // duplication did not perturb detect_token_loop.
    let mut t: Vec<u32> = (900u32..990).collect();
    for _ in 0..4 {
        t.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }
    assert!(
        detect_content_token_loop(&t),
        "exact byte-identical content loop must still be caught"
    );
}

#[test]
fn norm_fires_on_variable_length_digit_runs() {
    // The real shape: `- B(46) = 104509868777\n` — digit-level
    // tokenizer, so the index and value are RUNS of single-digit
    // tokens of DIFFERING length each line. Run-collapse must make
    // `- B(<run>) = <run>\n` identical regardless of digit count.
    let mask = numeric_mask();
    // structural template: [1,2,3] <idx-run> [4,5] <val-run> [6,7,8]
    // → collapsed period = 3 + 1 + 2 + 1 + 3 = 10 (>= PERIOD_MIN 8).
    let mut t: Vec<u32> = (900u32..990).collect();
    for k in 0..5u32 {
        t.extend([1, 2, 3]);
        // index run: 2..3 digit tokens, length varies with k
        t.extend(std::iter::repeat_n(100 + (k % 10), 2 + (k % 2) as usize));
        t.extend([4, 5]);
        // value run: 9..13 digit tokens, length varies with k
        t.extend(std::iter::repeat_n(101 + (k % 9), 9 + k as usize));
        t.extend([6, 7, 8]);
    }
    assert!(
        !detect_content_token_loop(&t),
        "exact detector misses: digit-run lengths differ every line"
    );
    assert!(
        detect_content_token_loop_normalized(&t, &mask),
        "run-collapse must catch the variable-length digit-run template"
    );
}

#[test]
fn norm_inert_with_empty_mask() {
    // mask=&[] → is_numeric always false → no sentinel → early-out.
    let t = varying_template_stream(5);
    assert!(
        !detect_content_token_loop_normalized(&t, &[]),
        "empty mask must make the normalized path inert (fail-open)"
    );
}
