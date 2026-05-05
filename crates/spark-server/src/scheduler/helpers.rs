// SPDX-License-Identifier: AGPL-3.0-only

//! Helpers: BF16 conversion, hard-stop registry, loop detection, sampling defaults.

/// Convert two little-endian BF16 bytes to f32.
#[inline]
pub fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
}

/// Global hard-stop token for ChatML role boundaries (`<|im_start|>`).
///
/// Set once at startup from `main.rs::set_im_start_hard_stop` when the
/// tokenizer exposes `<|im_start|>` as a single token id (Qwen3.5/3.6 family
/// tokenizers: id 248045). Read from `emit_token` to bail out of the turn
/// regardless of grammar / tool-call / min_tokens suppression — otherwise
/// the model can sample `<|im_start|>`, have it silently swallowed as a
/// suppressed EOS, and continue emitting the following role literal
/// (`user` / `assistant`, plain BPE tokens) which DO stream to the client.
///
/// 0 = unset / no hard-stop (non-Qwen tokenizers). The value is checked
/// with `load(Ordering::Relaxed)` on the emit path — no atomicity contract
/// beyond "set once before the first request lands", which is guaranteed
/// by the main.rs init ordering.
static IM_START_HARD_STOP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Install the ChatML role-boundary hard-stop. Called once from `main.rs`
/// at startup when `<|im_start|>` resolves to a single token id. Noop when
/// called with 0.
pub fn set_im_start_hard_stop(id: u32) {
    IM_START_HARD_STOP.store(id, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
pub fn im_start_hard_stop() -> Option<u32> {
    let id = IM_START_HARD_STOP.load(std::sync::atomic::Ordering::Relaxed);
    if id == 0 { None } else { Some(id) }
}
// ── Sampling defaults (SSOT) ────────────────────────────────────────────────
// All SamplingParams constructors reference these constants. Change here, not
// at each call site.
pub const DEFAULT_LZ_PENALTY: f32 = 0.0;
pub const DEFAULT_DRY_MULTIPLIER: f32 = 0.0;
pub const DEFAULT_DRY_BASE: f32 = 1.75;
// Was 2 (oobabooga's reference value, optimised for free-form text).
// Bumped to 3 (2026-04-25) because at allowed_length=2 the DRY sampler
// penalises legitimate code micro-repetition (consecutive `(`, `,`,
// indentation, two-line `let x =` patterns) and breaks tool-call JSON
// emission. allowed_length=3 still catches the bash-fence
// "Running: …Executing: …" attractor (which spans 6+ tokens) while
// letting normal source-code patterns through. Per Agent 8 SOTA
// research, this matches the consensus for code workloads.
pub const DEFAULT_DRY_ALLOWED_LENGTH: u32 = 3;

/// Token-level thinking-loop detection parameters. Tuned to catch
/// the Qwen3.5-35B-A3B fence-narration attractor (observed in dump
/// seq=19: `Running:\`\`\`bash cd X && cargo test\`\`\`Executing:
/// \`\`\`bash…\`\`\`…` cycling for the full 256-token thinking budget)
/// without false-positiving on legitimate numbered-list reasoning.
///
/// Strategy: once a sequence has spent `THINK_LOOP_MIN_TOKENS` inside
/// `<think>`, every `THINK_LOOP_CHECK_STRIDE` thinking tokens scan
/// the tail for a pattern of length `p ∈ [THINK_LOOP_PERIOD_MIN,
/// THINK_LOOP_PERIOD_MAX]` that repeats `THINK_LOOP_MIN_REPEATS`
/// times contiguously. If detected, set `force_end_thinking=true` so
/// the existing machinery force-emits `</think>` — the session
/// regains its full content budget instead of burning the thinking
/// cap. No workaround: attacks the phrase-loop attractor at its
/// earliest visible point, before it can monopolise the turn.
pub const THINK_LOOP_MIN_TOKENS: u32 = 48;
pub const THINK_LOOP_CHECK_STRIDE: u32 = 8;
pub const THINK_LOOP_PERIOD_MIN: usize = 4;
pub const THINK_LOOP_PERIOD_MAX: usize = 20;
pub const THINK_LOOP_MIN_REPEATS: usize = 3;
/// How many tokens back from the current tail to scan for needle
/// occurrences. Large enough to contain 3+ copies of a period-20
/// block (60 tokens) plus comfortable slack for the connective
/// prefixes that separate them.
pub const THINK_LOOP_SCAN_WINDOW: usize = 160;

/// Content-phase loop detection. Catches the post-`</think>` agentic
/// degeneration mode where the model emits the same sentence over
/// and over (observed 2026-04-26 against Claude Code: "I see I've
/// been creating Cargo.toml files but the user hasn't given me a
/// task. Let me wait for their instructions." × 12). LZ penalty
/// at strength 0.2 nudges but doesn't cure once the attractor is
/// established — we need a hard stop.
///
/// Periods extend up to 64 tokens because content-phase loops are
/// full sentences (20-50 tokens), not 4-20-token fence-narration
/// fragments. MIN_TOKENS is higher (96) to give legitimate prose
/// breathing room — three contiguous identical 30-token sentences
/// in a 280-token window is overwhelmingly degenerate.
///
/// Caveat: legitimate structured-code generation also produces
/// period-N repetition. Examples that false-positive:
/// - Chess board JS init: `{color:BLACK,type:'P'},` × 8 (period ~10)
/// - Arrays of identical empty-row HTML cells, multiplication
///   tables, JSON arrays of similar objects, repeated CSS rule
///   blocks, etc.
///
/// **Gating**: this watchdog is OFF by default. Models with a known
/// prose-attractor failure mode (Qwen3.5-35B-A3B + Claude-Code agentic
/// sessions) opt in via MODEL.toml `[behavior].enable_loop_watchdog =
/// true`. The flag is read at boot and stored in
/// [`set_enable_loop_watchdog`] / [`enable_loop_watchdog`].
pub const CONTENT_LOOP_MIN_TOKENS: u32 = 96;
pub const CONTENT_LOOP_CHECK_STRIDE: u32 = 16;
pub const CONTENT_LOOP_PERIOD_MIN: usize = 8;
pub const CONTENT_LOOP_PERIOD_MAX: usize = 64;
pub const CONTENT_LOOP_MIN_REPEATS: usize = 3;
pub const CONTENT_LOOP_SCAN_WINDOW: usize = 280;

static ENABLE_LOOP_WATCHDOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Set once at startup from the resolved `ModelBehavior.enable_loop_watchdog`.
/// Idempotent: subsequent calls within the same process are ignored.
pub fn set_enable_loop_watchdog(enabled: bool) {
    let _ = ENABLE_LOOP_WATCHDOG.set(enabled);
}

/// Read the per-model loop-watchdog flag set at boot. Defaults to
/// `false` until `set_enable_loop_watchdog` runs (boot order: weights →
/// behavior plumbing → scheduler start).
pub fn enable_loop_watchdog() -> bool {
    *ENABLE_LOOP_WATCHDOG.get().unwrap_or(&false)
}

/// F2 (2026-04-26): cap on free-text tokens between successive
/// `<tool_call>` opens when `tool_choice="auto"`. The grammar FSM
/// in `auto` mode (grammar.rs:461-462) sets `at_least_one=false`
/// and `stop_after_first=false`, so `is_terminated()` stays false
/// forever after the first tool call — the model can emit
/// prose↔tool↔prose↔tool indefinitely. 384 tokens is enough for
/// three normal "I'll now do X" paragraphs of agentic narrative;
/// anything beyond is the failure mode (re-narrating the plan
/// rather than executing it). Counted across non-thinking,
/// non-tool-body tokens only.
pub const MAX_INTER_TOOL_PROSE: u32 = 384;

/// F26 (2026-04-26): kernel-level entropy-collapse guard.
///
/// Disabled (`STREAK_K = 0`). Field experience: F26's pure-entropy
/// threshold can't distinguish wedged sampling from legitimate
/// high-confidence output (confident prose, code, JSON arrays). The
/// content-loop watchdog (`detect_content_token_loop`) gated on
/// per-model `enable_loop_watchdog` is the correct detector for actual
/// attractor states.
///
/// Constants kept so the call site at `decode_logits_seq.rs:285` still
/// type-checks; with `STREAK_K = 0` the gate is a no-op.
pub const ENTROPY_COLLAPSE_THRESHOLD_NATS: f32 = 0.5;
pub const ENTROPY_COLLAPSE_STREAK_K: u32 = 0;
pub const ENTROPY_COLLAPSE_WARMUP_TOKENS: usize = 32;

/// F27 (2026-04-26): logit-space attractor fingerprint.
///
/// Hash the f32_logits at 64 strided positions (~vocab/64 spacing).
/// Two tokens with near-identical logit distributions produce the
/// same fingerprint. If the same fingerprint repeats across recent
/// samples WHILE tokens varied (different sampled token each step),
/// the model is in an attractor where its decision space is stable
/// but it samples differently each time — exactly the
/// "different-tokens, same-internal-state" pattern hidden-state
/// cosine catches but at logit level (~1 µs/token, no kernel work).
///
/// `F27_RING_CAP`: ring buffer of recent fingerprints.
/// `F27_STREAK_K`: consecutive matches before tripping.
/// Same warmup + guard semantics as F26.
pub const F27_RING_CAP: usize = 16;
pub const F27_STREAK_K: u32 = 6;
pub const F27_FINGERPRINT_SAMPLES: usize = 64;
pub const F27_FINGERPRINT_QUANT: f32 = 2.0; // ~0.5 nat resolution

/// Compute a strided 64-bit fingerprint of the logit distribution.
/// Quantises each sampled value to ~0.5 nat resolution before
/// hashing so tiny FP-noise differences don't change the hash.
pub fn fingerprint_logits_strided(logits: &[f32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut h = DefaultHasher::new();
    let stride = (logits.len() / F27_FINGERPRINT_SAMPLES).max(1);
    let mut taken = 0;
    let mut i = 0;
    while i < logits.len() && taken < F27_FINGERPRINT_SAMPLES {
        let v = logits[i];
        let q: i32 = if v.is_finite() {
            (v * F27_FINGERPRINT_QUANT).round() as i32
        } else {
            i32::MIN
        };
        h.write_i32(q);
        i += stride;
        taken += 1;
    }
    h.finish()
}

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
    detect_token_loop(
        tokens,
        THINK_LOOP_MIN_TOKENS as usize,
        THINK_LOOP_PERIOD_MIN,
        THINK_LOOP_PERIOD_MAX,
        THINK_LOOP_MIN_REPEATS,
        THINK_LOOP_SCAN_WINDOW,
    )
}

/// Content-phase analogue of [`detect_thinking_token_loop`] — fires
/// when the model emits the same sentence over and over after
/// `</think>` has closed (the Claude-Code 2026-04-26 degeneration).
pub fn detect_content_token_loop(tokens: &[u32]) -> bool {
    detect_token_loop(
        tokens,
        CONTENT_LOOP_MIN_TOKENS as usize,
        CONTENT_LOOP_PERIOD_MIN,
        CONTENT_LOOP_PERIOD_MAX,
        CONTENT_LOOP_MIN_REPEATS,
        CONTENT_LOOP_SCAN_WINDOW,
    )
}

/// Substring-occurrence loop detector used by both the thinking
/// and content phases. Returns `true` iff some contiguous
/// subsequence of length `p ∈ [period_min, period_max]` appears
/// `min_repeats`+ times in the last `scan_window` tokens of `tokens`.
pub fn detect_token_loop(
    tokens: &[u32],
    min_tokens: usize,
    period_min: usize,
    period_max: usize,
    min_repeats: usize,
    scan_window: usize,
) -> bool {
    let n = tokens.len();
    if n < min_tokens {
        return false;
    }
    let tail_start = n.saturating_sub(scan_window);
    let tail = &tokens[tail_start..];
    for period in period_min..=period_max {
        if tail.len() < period * min_repeats {
            continue;
        }
        let needle = &tail[tail.len() - period..];
        let mut count = 0usize;
        let mut pos = 0usize;
        while pos + period <= tail.len() {
            if &tail[pos..pos + period] == needle {
                count += 1;
                if count >= min_repeats {
                    return true;
                }
                pos += period; // non-overlapping
            } else {
                pos += 1;
            }
        }
    }
    false
}
#[cfg(test)]
mod thinking_loop_tests {
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
}
