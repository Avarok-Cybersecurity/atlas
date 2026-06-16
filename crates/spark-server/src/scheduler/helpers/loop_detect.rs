// SPDX-License-Identifier: AGPL-3.0-only

//! Token-level loop detection (thinking-phase + content-phase, exact and
//! digit-normalized) and the shared vLLM-anchored period matcher. Extracted
//! from `helpers.rs` to keep that file ≤500 LoC. The thresholds and
//! `watchdog_params()` they read live in the parent `helpers` module.

use super::{
    CONTENT_LOOP_MIN_REPEATS, CONTENT_LOOP_MIN_TOKENS, CONTENT_LOOP_NORM_MIN_REPEATS,
    CONTENT_LOOP_PERIOD_MAX, CONTENT_LOOP_PERIOD_MIN, CONTENT_LOOP_SCAN_WINDOW, NUMERIC_SENTINEL,
    THINK_LOOP_MIN_TOKENS, THINK_LOOP_PERIOD_MAX, THINK_LOOP_PERIOD_MIN, watchdog_params,
};

/// Return `true` iff some contiguous subsequence of length
/// `p ∈ [THINK_LOOP_PERIOD_MIN, THINK_LOOP_PERIOD_MAX]` appears
/// `THINK_LOOP_MIN_REPEATS`+ times in the last
/// `THINK_LOOP_SCAN_WINDOW` tokens.
///
/// Designed to catch the Qwen3.5-35B fence-narration attractor where
/// the loop has a stable phrase body (` \`\`\`bash cd X && cargo test
/// \`\`\` `) but varying connective prefixes (`Running:` /
/// `Executing:` / `I need to run:`). A strict "contiguous
/// periodic repeat" detector misses these; a substring-occurrence
/// counter catches them.
pub fn detect_thinking_token_loop(tokens: &[u32]) -> bool {
    detect_thinking_token_loop_with(tokens, None)
}

/// Per-sequence override variant of [`detect_thinking_token_loop`].
/// When `override_` is `Some(p)`, uses `p.min_pattern_size`,
/// `p.max_pattern_size`, `p.min_count` as the period and repeat
/// thresholds — exactly mirroring vLLM's `RepetitionDetectionParams`
/// (`sampling_params.py:111-144`). When `None`, falls back to the
/// boot-global `watchdog_params()` constants so existing callers
/// without per-request configuration are byte-identical to before.
pub fn detect_thinking_token_loop_with(
    tokens: &[u32],
    override_: Option<crate::openai::RepetitionDetectionParams>,
) -> bool {
    let (period_min, period_max, min_repeats) = match override_ {
        Some(p) => (
            p.min_pattern_size as usize,
            p.max_pattern_size as usize,
            p.min_count as usize,
        ),
        None => {
            let wp = watchdog_params();
            (
                THINK_LOOP_PERIOD_MIN,
                THINK_LOOP_PERIOD_MAX,
                wp.think_loop_min_repeats,
            )
        }
    };
    let scan_window = match override_ {
        Some(_) => 0, // vLLM-anchored detector ignores scan_window
        None => watchdog_params().think_loop_scan_window,
    };
    detect_token_loop(
        tokens,
        THINK_LOOP_MIN_TOKENS as usize,
        period_min,
        period_max,
        min_repeats,
        scan_window,
    )
}

/// Content-phase analogue of [`detect_thinking_token_loop`] — fires
/// when the model emits the same sentence over and over after
/// `</think>` has closed (the Claude-Code 2026-04-26 degeneration).
pub fn detect_content_token_loop(tokens: &[u32]) -> bool {
    detect_content_token_loop_with(tokens, None)
}

/// Per-sequence override variant of [`detect_content_token_loop`].
/// `Some(p)` uses `p.min_pattern_size`, `p.max_pattern_size`,
/// `p.min_count`; `None` falls back to the historical content-loop
/// constants. See [`detect_thinking_token_loop_with`] for rationale.
pub fn detect_content_token_loop_with(
    tokens: &[u32],
    override_: Option<crate::openai::RepetitionDetectionParams>,
) -> bool {
    let (period_min, period_max, min_repeats) = match override_ {
        Some(p) => (
            p.min_pattern_size as usize,
            p.max_pattern_size as usize,
            p.min_count as usize,
        ),
        None => (
            CONTENT_LOOP_PERIOD_MIN,
            CONTENT_LOOP_PERIOD_MAX,
            CONTENT_LOOP_MIN_REPEATS,
        ),
    };
    detect_token_loop(
        tokens,
        CONTENT_LOOP_MIN_TOKENS as usize,
        period_min,
        period_max,
        min_repeats,
        CONTENT_LOOP_SCAN_WINDOW,
    )
}

/// Digit-normalized content-loop detector. Maps every numeric token in
/// the scan-window TAIL to [`NUMERIC_SENTINEL`], then period-matches —
/// catching the Qwen3.6-27B greedy degeneration where the line template
/// is fixed (`- B(46) = N\n`) but the integer payload varies each line,
/// so the exact [`detect_content_token_loop`] never fires.
///
/// Allocates only the ≤ `CONTENT_LOOP_SCAN_WINDOW` tail copy; the full
/// history is never normalized. FP mitigation: stricter
/// `CONTENT_LOOP_NORM_MIN_REPEATS`, and the matched period must contain
/// BOTH a sentinel (numeric) and a non-sentinel (structural) token —
/// pure-number columns and pure-prose loops are left to the exact path.
pub fn detect_content_token_loop_normalized(tokens: &[u32], mask: &[bool]) -> bool {
    detect_content_token_loop_normalized_with(tokens, mask, None)
}

/// Per-sequence override variant of
/// [`detect_content_token_loop_normalized`]. `Some(p)` substitutes the
/// caller's `(min_pattern_size, max_pattern_size, min_count)` for the
/// historical content-loop normalized constants. `None` preserves the
/// boot-global thresholds, matching the legacy call-site behaviour.
pub fn detect_content_token_loop_normalized_with(
    tokens: &[u32],
    mask: &[bool],
    override_: Option<crate::openai::RepetitionDetectionParams>,
) -> bool {
    let n = tokens.len();
    if n < CONTENT_LOOP_MIN_TOKENS as usize {
        return false;
    }
    let tail_start = n.saturating_sub(CONTENT_LOOP_SCAN_WINDOW);
    let is_numeric = |t: u32| (t as usize) < mask.len() && mask[t as usize];
    // Map numeric tokens to the sentinel AND run-length-collapse
    // consecutive sentinels to ONE. Qwen3.6 is digit-level
    // (`104509868777` → 12 single-digit tokens, `273508641` → 9), so a
    // bare 1:1 map would leave variable-length sentinel runs and the
    // period would still vary line to line. Collapsing makes
    // `- B(<digits>) = <digits>\n` identical regardless of digit count.
    let mut norm: Vec<u32> = Vec::with_capacity(CONTENT_LOOP_SCAN_WINDOW);
    for &t in &tokens[tail_start..] {
        if is_numeric(t) {
            if norm.last() != Some(&NUMERIC_SENTINEL) {
                norm.push(NUMERIC_SENTINEL);
            }
        } else {
            norm.push(t);
        }
    }
    // No qualifying period can exist without both kinds of token —
    // cheap early-out before the O(period·window) scan.
    let has_sentinel = norm.contains(&NUMERIC_SENTINEL);
    let has_struct = norm.iter().any(|&t| t != NUMERIC_SENTINEL);
    if !has_sentinel || !has_struct {
        return false;
    }
    let (period_min, period_max, min_repeats) = match override_ {
        Some(p) => (
            p.min_pattern_size as usize,
            p.max_pattern_size as usize,
            p.min_count as usize,
        ),
        None => (
            CONTENT_LOOP_PERIOD_MIN,
            CONTENT_LOOP_PERIOD_MAX,
            CONTENT_LOOP_NORM_MIN_REPEATS,
        ),
    };
    detect_token_loop_with_period(
        &norm,
        period_min,
        period_max,
        min_repeats,
        CONTENT_LOOP_SCAN_WINDOW,
    )
}

/// 2026-05-24 v3: ALGORITHM REPLACE. Switched from Atlas's scan-anywhere
/// substring detector to vLLM's anchored-at-end algorithm (vLLM main
/// `v1/core/sched/utils.py::_has_repeating_pattern`, GitHub
/// vllm-project/vllm; verified identical in 0.17.0 + current main).
///
/// **Why**: Atlas's scan-anywhere algorithm fires on ANY period match
/// in the last 280 tokens — including OLD patterns the model has
/// already moved past. Manifests as false-positive cutoffs on
/// numbered lists ("Step 1: Step 2: Step 3: Verify Cargo.toml" has
/// period-2 in the [Step,N] tail BEFORE the prose continuation, so
/// Atlas would fire even though the model is no longer looping).
///
/// **vLLM's algorithm**: take the LAST `pattern_len` tokens as a fixed
/// anchor; check whether the preceding `(min_repeats - 1)` windows of
/// the same length are byte-identical to it. If yes, the model is
/// CURRENTLY in a loop of period `pattern_len`. False positives on
/// historic patterns disappear because the check is end-anchored.
///
/// **`scan_window` kept for signature compat** — unused now, since the
/// vLLM algorithm only reads the last `pattern_len * min_repeats`
/// tokens (bounded automatically).
pub fn detect_token_loop(
    tokens: &[u32],
    min_tokens: usize,
    period_min: usize,
    period_max: usize,
    min_repeats: usize,
    _scan_window: usize,
) -> bool {
    let n = tokens.len();
    if n < min_tokens {
        return false;
    }
    if min_repeats < 2 {
        return false;
    }
    let period_min = period_min.max(1);
    for pattern_len in period_min..=period_max {
        if pattern_len * min_repeats > n {
            return false;
        }
        if has_repeating_pattern_anchored(tokens, pattern_len, min_repeats) {
            return true;
        }
    }
    false
}

/// vLLM-style anchored detector (port of
/// `vllm/v1/core/sched/utils.py::_has_repeating_pattern`). For each
/// position `n ∈ [1, pattern_len]` in the LAST `pattern_len` tokens,
/// verify that position is byte-identical at offsets
/// `pattern_len * m` (for m = 1..min_repeats) preceding the tail.
///
/// Caller MUST ensure `len(tokens) >= pattern_len * min_repeats`.
#[inline]
fn has_repeating_pattern_anchored(tokens: &[u32], pattern_len: usize, min_repeats: usize) -> bool {
    let n = tokens.len();
    for offset_in_window in 1..=pattern_len {
        let target = tokens[n - offset_in_window];
        for m in 1..min_repeats {
            let idx = n - (pattern_len * m + offset_in_window);
            if tokens[idx] != target {
                return false;
            }
        }
    }
    true
}

/// 2026-05-24 v3: vLLM-style anchored variant of the digit-normalized
/// detector. Same end-anchored check as [`detect_token_loop`] PLUS
/// the digit-normalized predicate: the matched window (last
/// `pattern_len` tokens) must contain BOTH a [`NUMERIC_SENTINEL`] and
/// a non-sentinel token. Without that mix, pure-number columns or
/// pure-prose loops would trip here (the exact detector's job).
fn detect_token_loop_with_period(
    tokens: &[u32],
    period_min: usize,
    period_max: usize,
    min_repeats: usize,
    _scan_window: usize,
) -> bool {
    let n = tokens.len();
    if min_repeats < 2 {
        return false;
    }
    let period_min = period_min.max(1);
    for pattern_len in period_min..=period_max {
        if pattern_len * min_repeats > n {
            return false;
        }
        let window = &tokens[n - pattern_len..];
        let has_numeric = window.contains(&NUMERIC_SENTINEL);
        let has_structural = window.iter().any(|&t| t != NUMERIC_SENTINEL);
        if !(has_numeric && has_structural) {
            continue;
        }
        if has_repeating_pattern_anchored(tokens, pattern_len, min_repeats) {
            return true;
        }
    }
    false
}
