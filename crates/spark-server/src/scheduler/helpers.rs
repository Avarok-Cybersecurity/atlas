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

/// Fix B (2026-06-05): global hard-stop token for the `<tool_response>` control
/// token. Set once at startup from `tokenizer_runtime.rs` when `<tool_response>`
/// resolves to a single token id; mirrors the `<|im_start|>` hard-stop above.
/// 0 = unset / no hard-stop.
static TOOL_RESPONSE_HARD_STOP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
pub fn set_tool_response_hard_stop(id: u32) {
    TOOL_RESPONSE_HARD_STOP.store(id, std::sync::atomic::Ordering::Relaxed);
}
pub fn tool_response_hard_stop() -> Option<u32> {
    let id = TOOL_RESPONSE_HARD_STOP.load(std::sync::atomic::Ordering::Relaxed);
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
// 2026-05-23 numerical-drift sweep lowered MIN_TOKENS 96→48 and
// MIN_REPEATS 3→2: opencode session ses_1a97c9241ffecMUu29IF8304TS
// showed the model entering a sentence-repeat attractor at late
// layers (MoE expert routing flipped at L38 due to ~7% accumulated
// drift, see project_qwen36_drift_moe_smoking_gun.md). With the old
// MIN_TOKENS=96 + MIN_REPEATS=3 thresholds the watchdog only armed
// AFTER 3 × ~16 tokens = ~48 tokens of identical-sentence repeats,
// PLUS a 96-token warm-up, so the attractor had already locked in
// and emitted hundreds of repeats. Halving both lets the watchdog
// fire within ~32 tokens of the second identical sentence, breaking
// the attractor before it stabilises.
pub const CONTENT_LOOP_MIN_TOKENS: u32 = 48;
pub const CONTENT_LOOP_CHECK_STRIDE: u32 = 16;
// 2026-05-24 sweep #2: MIN_REPEATS bumped 2 → 3 to match vLLM's
// `RepetitionDetectionParams.min_count` default. The earlier value of
// 2 was tuned for Atlas's pre-anchored substring-scan detector, where
// 4 tokens of matching tail was strong evidence of a loop. After the
// switch to vLLM's end-anchored algorithm (commit 1bb82ed), 2 repeats
// of period-2 (= 4 tokens) became a false-positive on legitimate
// JSON tool-call bodies — the structural `","` / `":"` punctuation
// tokens form a natural period-2 pattern. Observed live
// (opencode-phaseAB.jsonl 2026-05-24 18:13:18): watchdog fired at
// content_tokens=48 inside the bash tool-call body, prematurely
// ending the response (`reason=NoBoundary`, rollback declined).
// MIN_REPEATS=3 requires 6 consecutive end-anchored tokens — still
// fast on real `[A, B]` attractors (~100 ms after onset), but lets
// `","`-`":"` JSON noise pass.
//
// PERIOD_MIN stays at 2 to keep the tight `parameter>\nparameter>\n`
// real-loop case detectable (see project_qwen36_phase1_shipped note
// on the 21k-token hang). With MIN_REPEATS=3 that case requires 6
// matching tokens — still fires within the 64-token detection
// window the watchdog uses.
pub const CONTENT_LOOP_PERIOD_MIN: usize = 2;
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

/// Resolved kill-switch for ALL auto-watchdogs (content-loop, inter-tool
/// prose budget, F2 confidence early-stop, mid-word `</think>` defer,
/// thinking-loop). Cached once on first read from `ATLAS_DISABLE_WATCHDOGS`.
///
/// 2026-05-24: introduced for empirical test of whether Phase 2b
/// numerical fixes (RNE FP32 → BF16 + `__expf` softmax replacing the
/// 0.5 % polynomial) eliminated the degeneration that watchdogs catch.
/// Watchdogs were originally compensating for FP8 token-margin flips
/// pre-Phase 2b; better precision should reduce or eliminate the need.
///
/// `ATLAS_DISABLE_WATCHDOGS=1`/`true` (case-insensitive) → all
/// auto-watchdogs short-circuit. The user-set `max_thinking_budget` and
/// safety masks (post-`</think>` re-entry, tool-call-during-thinking)
/// are NOT touched — those are not watchdogs.
static DISABLE_WATCHDOGS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn parse_disable_watchdogs(env: Option<&str>) -> bool {
    match env {
        Some(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true")
        }
        None => false,
    }
}

/// Whether all auto-watchdogs are disabled at runtime. `false` by
/// default; flipped only when `ATLAS_DISABLE_WATCHDOGS=1`/`true`.
pub fn disable_watchdogs() -> bool {
    *DISABLE_WATCHDOGS.get_or_init(|| {
        parse_disable_watchdogs(std::env::var("ATLAS_DISABLE_WATCHDOGS").ok().as_deref())
    })
}

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

/// Resolve a "default-ON, explicit-disable" env flag. `None` (unset) → ON.
/// A falsy value (`"0"` / `"false"`, case-insensitive, trimmed) → OFF.
/// Everything else (`"1"`, `"true"`, junk) → ON. Mirrors the disable-idiom
/// of [`parse_forced_token_fastpath`].
fn env_flag_default_on(name: &str) -> bool {
    match std::env::var(name).ok().as_deref().map(str::trim) {
        Some(v) => !(v == "0" || v.eq_ignore_ascii_case("false")),
        None => true,
    }
}

/// Fix A (2026-06-05, baked default ON): lift EOS suppression after a
/// completed tool call in auto mode (is_terminated() never becomes true there).
/// This is the verified root-cause fix for the post-think cap-burn that drove
/// the webserver_ok 10/10 + Σwall win; default ON so the win is not env-dependent.
/// Kill-switch preserved: `ATLAS_TOOL_EOS_ESCAPE=0`/`false` disables.
pub fn tool_eos_escape_enabled() -> bool {
    env_flag_default_on("ATLAS_TOOL_EOS_ESCAPE")
}
/// #144 (budget-aware graceful close, default ON): when a length-limited
/// structured-output response would otherwise stop with the EOS token
/// grammar-illegal mid-structure (e.g. inside an open JSON string), emit the
/// shortest grammar-legal close so the truncated output is still parseable.
/// Kill-switch: `ATLAS_GRAMMAR_BUDGET_CLOSE=0`/`false` disables.
pub fn grammar_budget_close_enabled() -> bool {
    env_flag_default_on("ATLAS_GRAMMAR_BUDGET_CLOSE")
}
/// Fix B (2026-06-05, baked default ON): hard-stop on the <tool_response>
/// control token (a token the model must never generate). Default ON for the
/// same reason as Fix A. Kill-switch: `ATLAS_TOOL_RESPONSE_STOP=0`/`false`.
pub fn tool_response_stop_enabled() -> bool {
    env_flag_default_on("ATLAS_TOOL_RESPONSE_STOP")
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
    /// Default 60 (`CONFIDENCE_RUN_LIMIT`; 2026-05-23 sweep raised from 30).
    pub confidence_run_length: u32,
    /// Fuzzy-repetition detector Hamming tolerance divisor: a
    /// `pattern_len`-token window tolerates `pattern_len / div`
    /// mismatches. Default 12 (~8%).
    pub fuzzy_repeat_tolerance_div: usize,
    /// Cap on free-text tokens between successive `<tool_call>` opens in
    /// `tool_choice=auto`. Default 384 (`MAX_INTER_TOOL_PROSE`).
    pub max_inter_tool_prose: u32,
    /// Unconditional per-generation cap on post-`</think>` content tokens
    /// for tool-active requests. Default 100_000
    /// (`MAX_POST_THINK_CONTENT_TOKENS`) = effectively unbounded, the
    /// historical no-op. Backstops a grammar-legal-but-never-closing
    /// runaway that would otherwise burn to `max_tokens`.
    pub max_post_think_content_tokens: u32,
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
    max_post_think_content_tokens: MAX_POST_THINK_CONTENT_TOKENS,
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

/// Per-token mid-word mask (2026-05-24): `mask[id]` is true iff the
/// token decodes to text whose LAST character is alphanumeric — i.e.
/// emitting `</think>` (or any sentence-end punctuation) right after
/// this token would split a word.
///
/// Used by [`super::decode_logits_seq`] to suppress `</think>` when the
/// previously emitted token ended mid-word. FP8 precision drift on
/// Qwen3.6-FP8 biases the `</think>` logit upward by enough to flip
/// against word-continuation tokens at low margin (opencode-session.md
/// 2026-05-24: 8/8 thinking blocks ended mid-word: "creating thep",
/// "ping/pong en", "then cr"). The fix is a soft guard rather than a
/// rewrite of the model.
///
/// Fail-open: never set → suppression is skipped and the model
/// retains full freedom to terminate thinking at any token.
static MID_WORD_TOKEN_MASK: std::sync::OnceLock<std::sync::Arc<[bool]>> =
    std::sync::OnceLock::new();

/// Set once at startup from the resolved tokenizer. Idempotent.
pub fn set_mid_word_token_mask(mask: std::sync::Arc<[bool]>) {
    let _ = MID_WORD_TOKEN_MASK.set(mask);
}

/// Read the mid-word token mask. `None` until `set_mid_word_token_mask`
/// runs — callers must treat `None` as "no mid-word info available"
/// and skip the suppression.
pub fn mid_word_token_mask() -> Option<std::sync::Arc<[bool]>> {
    MID_WORD_TOKEN_MASK.get().cloned()
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

/// F1 (2026-06-02): unconditional per-generation cap on post-`</think>`
/// content tokens for tool-active requests (`grammar_state.is_some()`).
/// The SSOT in-code default — 100_000, effectively unbounded — reproduces
/// the historical no-op so a model that sets nothing in MODEL.toml
/// `[behavior].max_post_think_content_tokens` is byte-identical to before.
/// A per-model value (e.g. 1536 on Qwen3.6-35B-A3B-FP8) backstops the
/// grammar-legal-but-never-closing tool-value runaway — a garbled/merged
/// BPE close token leaves the value rule unterminated, so the generation
/// burns to `max_tokens` and starves the agent's wall-clock budget. The
/// cap fires regardless of which `inside_tool_body` state machine
/// desynced; the `grammar_state.is_some()` gate ensures plain chat
/// (which attaches no grammar) is never truncated.
pub const MAX_POST_THINK_CONTENT_TOKENS: u32 = 100_000;

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
/// period-2 in the `[Step,N]` tail BEFORE the prose continuation, so
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

// F2 confidence-run + code-fence pure helpers (`toggle_code_fence`,
// `confidence_run_step`, `should_inject_think_end` + their constants)
// were moved to `confidence.rs` to keep this file ≤500 LoC. They are
// re-exported through the scheduler module so existing `super::*`
// call sites are unaffected.

#[cfg(test)]
#[path = "helpers_tests.rs"]
mod thinking_loop_tests;
