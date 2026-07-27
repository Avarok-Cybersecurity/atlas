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
fn rejects_fence_body_with_varying_prefixes() {
    // 2026-05-24: this case was previously detected by Atlas's
    // scan-anywhere substring-repeat detector. After the switch to
    // vLLM's end-anchored algorithm, the detector intentionally
    // does NOT fire here — the varying connective prefixes mean
    // no fixed period repeats at the buffer's END. This case is
    // now caught one layer up by the rollback-to-boundary +
    // re-steer machinery once a tighter end-anchored period
    // forms after the boundary rewind. Keeping the test as a
    // negative assertion to pin the contract.
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
        !detect_thinking_token_loop(&tokens),
        "end-anchored detector intentionally does not fire on varying-prefix patterns"
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
    // breathing room. Constants threshold-tracked via the named
    // constant so the test stays valid across the 2026-05-23
    // sweep (MIN_TOKENS 96→48).
    let pat: Vec<u32> = (1..=10).collect();
    // Build a response of (MIN_TOKENS - 4) total so we're below
    // the floor even after 3× repeats.
    let prior_len = (CONTENT_LOOP_MIN_TOKENS as usize).saturating_sub(3 * pat.len() + 4);
    let mut tokens: Vec<u32> = (50u32..50 + prior_len as u32).collect();
    tokens.extend(pat.iter());
    tokens.extend(pat.iter());
    tokens.extend(pat.iter());
    assert!(
        tokens.len() < CONTENT_LOOP_MIN_TOKENS as usize,
        "test setup error: tokens.len()={} exceeds MIN_TOKENS={}",
        tokens.len(),
        CONTENT_LOOP_MIN_TOKENS,
    );
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
fn content_loop_accepts_three_repeats() {
    // 2026-05-24 sweep #2: CONTENT_LOOP_MIN_REPEATS bumped 2 → 3 to
    // match vLLM's `RepetitionDetectionParams.min_count` default.
    // The 2-repeat firing was too aggressive — fired on legitimate
    // JSON tool-body period-2 punctuation (`","`/`":"`) and ended
    // responses mid-emission (opencode-phaseAB.jsonl 2026-05-24
    // 18:13:18: watchdog matched period=2 repeats=2 at
    // content_tokens=48, prematurely ending a bash tool call).
    // Three byte-exact repeats of a 30-token block remain a
    // genuine sentence-attractor and MUST trigger.
    let sentence: Vec<u32> = (500..530).collect();
    let mut tokens: Vec<u32> = (0..100).collect();
    tokens.extend(sentence.iter()); // r1
    tokens.extend(sentence.iter()); // r2
    tokens.extend(sentence.iter()); // r3
    assert!(
        detect_content_token_loop(&tokens),
        "three byte-exact 30-token repeats must trigger watchdog at MIN_REPEATS={}",
        CONTENT_LOOP_MIN_REPEATS,
    );
}

#[test]
fn content_loop_rejects_two_repeats() {
    // Companion to `accepts_three_repeats`: at the new MIN_REPEATS=3
    // default, two byte-exact copies are NOT enough evidence — a
    // legitimate "I said X. I said X again." pattern (or JSON
    // structural punctuation tokens repeating once) is below the
    // firing bar. The fuzzy / DRY / LZ samplers cover that band.
    let sentence: Vec<u32> = (500..530).collect();
    let mut tokens: Vec<u32> = (0..100).collect();
    tokens.extend(sentence.iter()); // r1
    tokens.extend(sentence.iter()); // r2 only
    assert!(
        !detect_content_token_loop(&tokens),
        "two byte-exact 30-token repeats must NOT trigger at MIN_REPEATS={}",
        CONTENT_LOOP_MIN_REPEATS,
    );
}

#[test]
fn content_loop_rejects_single_occurrence() {
    // Single occurrence of any pattern (no repeats) must NOT
    // trigger. Replaces the prior `two_repeats` rejection test
    // which was invalidated when MIN_REPEATS was lowered to 2.
    let sentence: Vec<u32> = (500..530).collect();
    let mut tokens: Vec<u32> = (0..100).collect();
    tokens.extend(sentence.iter()); // 1 occurrence — must not trigger
    assert!(
        !detect_content_token_loop(&tokens),
        "single occurrence (no repeat) must not trigger watchdog"
    );
}

// F2 confidence-run + code-fence tests moved to `confidence_tests.rs`
// alongside the helpers themselves (`confidence.rs`).

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

// ── Per-request RepetitionDetectionParams override tests ─────────────

#[test]
fn override_loosens_content_loop_threshold() {
    // Three contiguous copies of a 22-token sentence — passes the
    // boot-default `CONTENT_LOOP_MIN_REPEATS=3` so the default
    // detector fires. With a stricter `min_count=4` override, the
    // detector must NOT fire on the same input. This proves the
    // override actually wins over the boot default.
    let sentence: Vec<u32> = (1000..1022).collect();
    let mut tokens: Vec<u32> = (0..100).collect(); // prior content
    tokens.extend(sentence.iter()); // r1
    tokens.extend(sentence.iter()); // r2
    tokens.extend(sentence.iter()); // r3

    // Default path: 3 repeats at period 22, MIN_REPEATS=3 ⇒ fires.
    assert!(
        detect_content_token_loop_with(&tokens, None),
        "default thresholds must still fire on 22-token × 3 repeat"
    );

    // Override path: min_count=4 ⇒ 3 repeats are insufficient.
    let strict = crate::api::inference_types::RepetitionDetectionParams {
        min_pattern_size: 2,
        max_pattern_size: 64,
        min_count: 4,
    };
    assert!(
        !detect_content_token_loop_with(&tokens, Some(strict)),
        "stricter min_count=4 override must suppress 3-repeat firing"
    );
}

#[test]
fn override_tightens_content_loop_threshold() {
    // Five contiguous copies of a 5-token block. Below the boot-default
    // CONTENT_LOOP_MIN_TOKENS the detector won't even consider firing,
    // so pad with prior content first. With period_min=5 .. period_max=5
    // + min_count=3 the override fires on (5 × 5 = 25) end-anchored
    // tokens — covered by the 5-repeat tail.
    let pat: Vec<u32> = vec![42, 43, 44, 45, 46];
    let mut tokens: Vec<u32> = (0u32..50).collect();
    for _ in 0..5 {
        tokens.extend(pat.iter());
    }
    let permissive = crate::api::inference_types::RepetitionDetectionParams {
        min_pattern_size: 5,
        max_pattern_size: 5,
        min_count: 3,
    };
    assert!(
        detect_content_token_loop_with(&tokens, Some(permissive)),
        "override (period=5, min_count=3) must catch 5×period-5 tail"
    );
}

#[test]
fn override_applies_to_thinking_loop() {
    // 4× period-10 fence loop — fires under boot default
    // THINK_LOOP_MIN_REPEATS=3.
    let pat: Vec<u32> = vec![7, 6, 5, 4, 3, 2, 1, 0, 9, 8];
    let mut tokens: Vec<u32> = (100u32..150).collect();
    for _ in 0..4 {
        tokens.extend(pat.iter());
    }
    assert!(
        detect_thinking_token_loop_with(&tokens, None),
        "default thinking-loop thresholds must still fire on 4× period-10"
    );
    // Override demanding 6 repeats ⇒ 4 is insufficient ⇒ must not fire.
    let strict = crate::api::inference_types::RepetitionDetectionParams {
        min_pattern_size: 4,
        max_pattern_size: 20,
        min_count: 6,
    };
    assert!(
        !detect_thinking_token_loop_with(&tokens, Some(strict)),
        "stricter min_count=6 override must suppress 4-repeat firing"
    );
}

// ── Forced-token fast-path kill-switch parsing ──────────────────────────────

#[test]
fn forced_token_fastpath_default_enabled() {
    // Env unset → fast-path on (the default; output is bit-identical to
    // the sampled path so there is no reason to ship it off).
    assert!(parse_forced_token_fastpath(None));
}

#[test]
fn forced_token_fastpath_disabled_by_truthy() {
    // Explicit truthy values disable the fast-path (the kill-switch).
    assert!(!parse_forced_token_fastpath(Some("1")));
    assert!(!parse_forced_token_fastpath(Some("true")));
    assert!(!parse_forced_token_fastpath(Some("TRUE")));
    assert!(!parse_forced_token_fastpath(Some("  true  ")));
}

#[test]
fn forced_token_fastpath_enabled_by_falsy_or_junk() {
    // Anything that is not an explicit truthy value keeps it enabled —
    // `0`, `false`, empty, and junk all mean "do not disable".
    assert!(parse_forced_token_fastpath(Some("0")));
    assert!(parse_forced_token_fastpath(Some("false")));
    assert!(parse_forced_token_fastpath(Some("")));
    assert!(parse_forced_token_fastpath(Some("yes")));
}
