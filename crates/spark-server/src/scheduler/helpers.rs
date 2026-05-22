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
/// Min repeats for the digit-normalized content-loop path. Stricter
/// than `CONTENT_LOOP_MIN_REPEATS` (3) because numeric normalization
/// collapses more sequences to a common period — requiring 4 keeps a
/// legitimate 3-item numbered list (`- item 1\n- item 2\n- item 3`)
/// from tripping the hard stop.
pub const CONTENT_LOOP_NORM_MIN_REPEATS: usize = 4;
/// Sentinel substituted for every numeric token in the normalized
/// scan-window tail. `u32::MAX` can never collide with a real vocab id
/// (Qwen3.6 vocab ≤ ~152k), and the `(t as usize) < mask.len()` bound
/// in the classifier means a stray real `u32::MAX` would degrade to
/// "structural", never a false numeric — safe either way.
pub const NUMERIC_SENTINEL: u32 = u32::MAX;

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

// ── Grammar forced-token fast-path (xgrammar Tier 3b) ───────────────────────

/// Resolved kill-switch for the grammar forced-token (Coalescence)
/// fast-path. Computed once on first read from the environment.
///
/// The fast-path emits a token directly — skipping the model sample and
/// the vocab-wide bitmask fill — only when the active tool-call grammar
/// admits exactly one legal next token (xgrammar's `forced_token`
/// guarantees a single-bit mask). Output is therefore bit-identical to
/// the sampled path, so the fast-path is **on by default**.
///
/// `ATLAS_DISABLE_FORCED_TOKEN=1` (or `true`) forces it off — a
/// kill-switch should a future grammar/matcher regression ever make the
/// forced-token guarantee unsafe. This mirrors the env-var bisection
/// gates already used in `phase_continue_prefills.rs` /
/// `mod_helpers.rs`; a MODEL.toml `[behavior]` flag was not used because
/// the `ModelBehavior` struct lives in the `atlas-kernels` crate, which
/// this change deliberately does not touch.
static FORCED_TOKEN_FASTPATH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Pure parse of the `ATLAS_DISABLE_FORCED_TOKEN` env value into the
/// resolved "fast-path enabled" boolean. Split out of
/// [`forced_token_fastpath_enabled`] so the parsing rule is unit-testable
/// without touching the process-wide `OnceLock`.
///
/// `None` (env unset) → enabled. A truthy value (`"1"` / `"true"`,
/// case-insensitive, surrounding whitespace ignored) → disabled.
/// Everything else (empty, `"0"`, `"false"`, junk) → enabled.
fn parse_forced_token_fastpath(env: Option<&str>) -> bool {
    match env {
        Some(v) => {
            let v = v.trim();
            !(v == "1" || v.eq_ignore_ascii_case("true"))
        }
        None => true,
    }
}

/// Whether the grammar forced-token fast-path is enabled (default
/// `true`; disabled by `ATLAS_DISABLE_FORCED_TOKEN=1`/`true`).
pub fn forced_token_fastpath_enabled() -> bool {
    *FORCED_TOKEN_FASTPATH.get_or_init(|| {
        parse_forced_token_fastpath(std::env::var("ATLAS_DISABLE_FORCED_TOKEN").ok().as_deref())
    })
}

/// Per-model tunables for the always-on decode-time watchdogs. Sourced
/// from MODEL.toml `[behavior]`; the field defaults reproduce the
/// historical hardcoded constants exactly, so a model that sets nothing
/// behaves byte-identically to before parameterization.
#[derive(Debug, Clone, Copy)]
pub struct WatchdogParams {
    /// Thinking-loop watchdog: substring-occurrence count that trips a
    /// forced `</think>`. Default 3 (`THINK_LOOP_MIN_REPEATS`).
    pub think_loop_min_repeats: usize,
    /// Thinking-loop watchdog: trailing-token scan window. Default 160
    /// (`THINK_LOOP_SCAN_WINDOW`).
    pub think_loop_scan_window: usize,
    /// F2 confidence-run early-stop enabled. Default `true`. Set false in
    /// MODEL.toml for models whose deterministic code drafting trips the
    /// heuristic.
    pub confidence_early_stop: bool,
    /// F2 confidence run length before arming forced `</think>`.
    /// Default 30 (`CONFIDENCE_RUN_LIMIT`).
    pub confidence_run_length: u32,
    /// Fuzzy-repetition detector Hamming tolerance divisor: a
    /// `pattern_len`-token window tolerates `pattern_len / div`
    /// mismatches. Default 12 (~8%).
    pub fuzzy_repeat_tolerance_div: usize,
    /// Cap on free-text tokens between successive `<tool_call>` opens in
    /// `tool_choice=auto`. Default 384 (`MAX_INTER_TOOL_PROSE`).
    pub max_inter_tool_prose: u32,
    /// Phase-C: when a degeneration watchdog fires, roll back to the last
    /// well-formed boundary and re-steer instead of hard-stopping.
    /// Default `true`. See [`super::rollback::rollback_to_boundary`].
    pub rollback_resteer: bool,
}

/// Historical-default watchdog tunables — the single source of truth.
/// Each field equals the constant the watchdog used before
/// parameterization, so an unset MODEL.toml `[behavior]` is byte-exact.
/// `CONFIDENCE_RUN_LIMIT` now lives in the sibling `confidence` module
/// (F2 helper extraction); referenced here as the historical default.
const DEFAULT_WATCHDOG_PARAMS: WatchdogParams = WatchdogParams {
    think_loop_min_repeats: THINK_LOOP_MIN_REPEATS,
    think_loop_scan_window: THINK_LOOP_SCAN_WINDOW,
    confidence_early_stop: true,
    confidence_run_length: super::confidence::CONFIDENCE_RUN_LIMIT,
    fuzzy_repeat_tolerance_div: 12,
    max_inter_tool_prose: MAX_INTER_TOOL_PROSE,
    rollback_resteer: true,
};

impl Default for WatchdogParams {
    fn default() -> Self {
        DEFAULT_WATCHDOG_PARAMS
    }
}

static WATCHDOG_PARAMS: std::sync::OnceLock<WatchdogParams> = std::sync::OnceLock::new();

/// Set once at startup from the resolved `ModelBehavior`. Idempotent.
pub fn set_watchdog_params(p: WatchdogParams) {
    let _ = WATCHDOG_PARAMS.set(p);
}

/// Read the per-model watchdog tunables. Returns the historical-default
/// `WatchdogParams` until `set_watchdog_params` runs — so unit tests and
/// any pre-boot caller see exactly the old hardcoded constants.
pub fn watchdog_params() -> WatchdogParams {
    *WATCHDOG_PARAMS.get().unwrap_or(&DEFAULT_WATCHDOG_PARAMS)
}

/// `mask[id] == true` iff token `id` decodes to a pure ASCII-digit run
/// (optionally one leading space). Built once at startup from the
/// tokenizer; drives the digit-normalized content-loop path. Fail-open:
/// never set (or build failed) → normalized path inert, the exact
/// detector is unaffected.
static NUMERIC_TOKEN_MASK: std::sync::OnceLock<std::sync::Arc<[bool]>> = std::sync::OnceLock::new();

/// Set once at startup from the resolved tokenizer. Idempotent.
pub fn set_numeric_token_mask(mask: std::sync::Arc<[bool]>) {
    let _ = NUMERIC_TOKEN_MASK.set(mask);
}

/// Read the numeric-token mask. `None` until `set_numeric_token_mask`
/// runs — callers must treat `None` as "normalized path disabled".
pub fn numeric_token_mask() -> Option<std::sync::Arc<[bool]>> {
    NUMERIC_TOKEN_MASK.get().cloned()
}

/// `mask[id] == true` iff token `id` decodes to text ending in a
/// well-formed generation boundary — a newline, or sentence-ending
/// punctuation (`.`, `!`, `?`) optionally followed by a closing quote
/// or whitespace. Built once at startup from the tokenizer; drives
/// [`super::rollback::rollback_to_boundary`]'s boundary search.
/// Fail-open: never set → rollback finds no boundary and the watchdog
/// falls back to its hard stop.
static BOUNDARY_TOKEN_MASK: std::sync::OnceLock<std::sync::Arc<[bool]>> =
    std::sync::OnceLock::new();

/// Set once at startup from the resolved tokenizer. Idempotent.
pub fn set_boundary_token_mask(mask: std::sync::Arc<[bool]>) {
    let _ = BOUNDARY_TOKEN_MASK.set(mask);
}

/// Read the boundary-token mask. `None` until `set_boundary_token_mask`
/// runs — callers must treat `None` as "no boundary info available".
pub fn boundary_token_mask() -> Option<std::sync::Arc<[bool]>> {
    BOUNDARY_TOKEN_MASK.get().cloned()
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
    let wp = watchdog_params();
    detect_token_loop(
        tokens,
        THINK_LOOP_MIN_TOKENS as usize,
        THINK_LOOP_PERIOD_MIN,
        THINK_LOOP_PERIOD_MAX,
        wp.think_loop_min_repeats,
        wp.think_loop_scan_window,
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
    detect_token_loop_with_period(
        &norm,
        CONTENT_LOOP_PERIOD_MIN,
        CONTENT_LOOP_PERIOD_MAX,
        CONTENT_LOOP_NORM_MIN_REPEATS,
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

/// Like [`detect_token_loop`] but only accepts a periodic match whose
/// `needle` (the period-length window) contains BOTH a
/// [`NUMERIC_SENTINEL`] and a non-sentinel token. Used exclusively by
/// the digit-normalized path so a pure-number column or a pure-prose
/// repeat does not trip here (those remain the exact detector's job).
/// The `< min_tokens` guard is the caller's `CONTENT_LOOP_MIN_TOKENS`
/// check, so this takes only the tail/period rules.
///
/// ~20 lines duplicate `detect_token_loop`'s scan: the per-period
/// needle predicate needs the matched window, which `detect_token_loop`
/// (`-> bool`) hides. Duplicating is lower-risk than adding a closure
/// param to a function with 13 existing tests + the exact call site;
/// the exact detector stays byte-identical (regression-tested).
fn detect_token_loop_with_period(
    tokens: &[u32],
    period_min: usize,
    period_max: usize,
    min_repeats: usize,
    scan_window: usize,
) -> bool {
    let n = tokens.len();
    let tail_start = n.saturating_sub(scan_window);
    let tail = &tokens[tail_start..];
    for period in period_min..=period_max {
        if tail.len() < period * min_repeats {
            continue;
        }
        let needle = &tail[tail.len() - period..];
        let needle_ok =
            needle.contains(&NUMERIC_SENTINEL) && needle.iter().any(|&t| t != NUMERIC_SENTINEL);
        if !needle_ok {
            continue;
        }
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

// F2 confidence-run + code-fence pure helpers (`toggle_code_fence`,
// `confidence_run_step`, `should_inject_think_end` + their constants)
// were moved to `confidence.rs` to keep this file ≤500 LoC. They are
// re-exported through the scheduler module so existing `super::*`
// call sites are unaffected.

#[cfg(test)]
#[path = "helpers_tests.rs"]
mod thinking_loop_tests;
