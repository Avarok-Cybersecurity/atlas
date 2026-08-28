// SPDX-License-Identifier: AGPL-3.0-only

//! Helpers: BF16 conversion, loop detection, sampling defaults, and the pure
//! decision cores behind the hard output/length limits (the limits themselves
//! travel on `SchedCtx` — see `scheduler::limits`).

/// Convert two little-endian BF16 bytes to f32.
#[inline]
pub fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
}

// The `<|im_start|>` / `<tool_response>` hard-stop token ids and the
// served-context ceiling were three atomics here, each installed by a `set_*`
// call during serve startup. All three are per-model — two are token ids,
// which mean nothing against another tokenizer — so they are now
// `SchedLimits`, built where the tokenizer resolves them and carried on
// `SchedCtx`. See `scheduler::limits`.

// ── Hard output/length limits (2026-07-21, DS4F hard-limit lane) ─────────────
// Three scheduler defects let generation run past its declared ceilings once a
// long `<think>` block engaged (R1X overrun: past both max_tokens=4096 AND
// max_seq_len=8192):
//   C-1 thinking tokens never decremented the completion budget,
//   C-2 EOS was suppressed for the whole thinking span,
//   C-3 max_seq_len was only a KV-allocation ceiling trued-up on completion,
//       never enforced per decode step.
// The pure decision cores below back the fixes in `decode_logits_step` and
// `emit_step`. All are no-ops until a ceiling is actually reached, so the
// 35/40 direct-mode (thinking-OFF) baseline is byte-unchanged.

/// §C-3 pure core: would the NEXT decode step reach or exceed the served
/// context ceiling? `position` = current sequence length (prompt + generated,
/// `SequenceState.seq_len`). `max_seq_len == 0` means unset/disabled → never
/// fires. Fires one token BEFORE the ceiling (`position + 1 >= max_seq_len`,
/// per the handoff §E-3) so the sequence never writes KV at `max_seq_len`.
#[inline]
pub fn seqlen_force_stop(position: usize, max_seq_len: usize) -> bool {
    max_seq_len != 0 && position + 1 >= max_seq_len
}

/// §C-1/§C-2 pure core: has this sequence hit a HARD ceiling — completion
/// budget exhausted (`remaining == 0`) OR context ceiling reached? At a hard
/// ceiling the sequence MUST finish regardless of `inside_thinking`.
#[inline]
pub fn hard_ceiling_hit(remaining: usize, position: usize, max_seq_len: usize) -> bool {
    remaining == 0 || seqlen_force_stop(position, max_seq_len)
}

/// §C-2 pure core: suppress EOS inside a `<think>` block ONLY while no hard
/// ceiling is hit. `<|im_end|>` inside `<think>` is normally spurious (only
/// `</think>` exits thinking), but at the budget / seq-len ceiling a
/// model-sampled EOS MUST be honored so generation cannot run past its declared
/// limits. Identical to the old bare `inside_thinking` gate until a ceiling is
/// actually reached → preserves the direct-mode baseline.
#[inline]
pub fn eos_suppressed_by_thinking(inside_thinking: bool, hard_ceiling_hit: bool) -> bool {
    inside_thinking && !hard_ceiling_hit
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
/// true`. The flag is read at boot into `SchedLevers::loop_watchdog`, which
/// the dashboard's `/watchdog` command can also toggle mid-run.
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

// `ATLAS_DISABLE_WATCHDOGS` is resolved once into `SchedLevers::disable_watchdogs`
// and read through `SchedCtx` / `LogitsContext`. `parse_disable_watchdogs` below
// stays as the SSOT parse — the boot audit calls it directly, having no carrier.

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
pub(crate) fn parse_disable_watchdogs(env: Option<&str>) -> bool {
    match env {
        Some(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true")
        }
        None => false,
    }
}
// The loop watchdog was a `OnceLock<bool>` with a `set_` installer called
// from both serve startup (MODEL.toml `[behavior].enable_loop_watchdog`) and
// the dashboard's `/watchdog` command — a process global precisely because
// two threads needed to share one bool. It is now `SchedLevers::loop_watchdog`,
// an atomic inside the run's levers, which serve hands to the scheduler and
// the dashboard as an `Arc`.

// `enable_think_loop_watchdog` was a `OnceLock<bool>` with a setter called from
// `serve_phases::runtime` — the exact anti-pattern `WatchdogParams` documents
// below and was created to retire: the value is PER MODEL, so a process global
// keeps the first model's answer for every model loaded after it, and the
// installer ran after the scheduler thread had already spawned. It now lives in
// `WatchdogParams::enable_think_loop_watchdog`, built per load and carried.

// ── Grammar forced-token fast-path (xgrammar Tier 3b) ───────────────────────

// Resolved once into `SchedLevers::forced_token_fastpath` and read off
// `LogitsContext::sampling` by the one stage that needs it.
// `parse_forced_token_fastpath` below stays as the SSOT parse.

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
/// Pure parse of the `ATLAS_DISABLE_FORCED_TOKEN` env value into the
/// resolved "fast-path enabled" boolean. Kept separate from
/// `SchedLevers::forced_token_fastpath`, which calls it, so the parsing rule
/// is unit-testable without building a whole lever set.
///
/// `None` (env unset) → enabled. A truthy value (`"1"` / `"true"`,
/// case-insensitive, surrounding whitespace ignored) → disabled.
/// Everything else (empty, `"0"`, `"false"`, junk) → enabled.
pub(crate) fn parse_forced_token_fastpath(env: Option<&str>) -> bool {
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
    /// `tool_choice=auto`. Default [`MAX_INTER_TOOL_PROSE`]; `u32::MAX`
    /// here means an operator disabled the guard (see
    /// [`resolve_max_inter_tool_prose`]).
    pub max_inter_tool_prose: u32,
    /// Unconditional per-generation cap on post-`</think>` content tokens
    /// for tool-active requests. Default 100_000
    /// (`MAX_POST_THINK_CONTENT_TOKENS`) = effectively unbounded, the
    /// historical no-op. Backstops a grammar-legal-but-never-closing
    /// runaway that would otherwise burn to `max_tokens`.
    pub max_post_think_content_tokens: u32,
    /// Honor a mid-`<think>` EOS by implicitly closing the thinking block.
    /// Per-model (MODEL.toml `[behavior].honor_eos_inside_thinking`), default
    /// FALSE = pre-p350 behaviour (discard the EOS, let the model recover and
    /// emit its tool call). Carried here rather than in a `OnceLock` for the
    /// reason `from_behavior` documents below: a process global keeps the
    /// FIRST model's value for every model after it, and its installer runs
    /// after the scheduler thread is already spawned.
    pub honor_eos_inside_thinking: bool,
    /// Per-model gate for the THINKING-phase token-loop watchdog. Default
    /// `true` (the behaviour before the flag existed), so a model that does not
    /// opt out is unaffected. Was a `OnceLock`; see the note above.
    ///
    /// Why a model would opt out: the watchdog force-injects `</think>` on a
    /// period-4..20 repeat in the reasoning tail. When it misfires, the model
    /// is yanked out of a reasoning block it did not choose to end, has no
    /// natural continuation, and the post-close CONTENT degenerates into token
    /// spam. Observed on Laguna-S-2.1: watchdog on -> `finish=length` with
    /// content '####...' / backtick spam; same prompt with watchdogs off ->
    /// coherent reasoning and no spam.
    pub enable_think_loop_watchdog: bool,
    /// Phase-C: when a degeneration watchdog fires, roll back to the last
    /// well-formed boundary and re-steer instead of hard-stopping.
    /// Default `true`. See [`super::rollback::rollback_to_boundary`].
    pub rollback_resteer: bool,
    /// Operator override for the content-loop detector's repeat threshold
    /// (`--content-loop-min-repeats` / `ATLAS_CONTENT_LOOP_MIN_REPEATS`).
    /// `None` = the built-in [`CONTENT_LOOP_MIN_REPEATS`] (3). A
    /// per-request `repetition_detection` object still outranks this.
    /// Raising it loosens the guard for output whose legitimate shape is
    /// short-period repetition (code: `}\n}\n`, `,\n` list tails).
    pub content_loop_min_repeats: Option<u32>,
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
    // FALSE = pre-p350 behaviour: a mid-think EOS is discarded, not honored.
    honor_eos_inside_thinking: false,
    enable_think_loop_watchdog: true,
    content_loop_min_repeats: None,
};

impl Default for WatchdogParams {
    fn default() -> Self {
        DEFAULT_WATCHDOG_PARAMS
    }
}

impl WatchdogParams {
    /// Historical fuzzy-repeat mismatch tolerance divisor, exposed so the
    /// `repetition` tests can name it instead of restating `12`.
    pub const DEFAULT_FUZZY_TOLERANCE_DIV: usize =
        DEFAULT_WATCHDOG_PARAMS.fuzzy_repeat_tolerance_div;

    /// Resolve this model's watchdog tunables from its MODEL.toml
    /// `[behavior]` table, then the overrides that outrank it.
    ///
    /// `max_inter_tool_prose_cli` is `--max-inter-tool-prose` (#328); see
    /// [`resolve_max_inter_tool_prose`] for the precedence chain.
    /// `content_loop_min_repeats_cli` is `--content-loop-min-repeats`
    /// (#328 family); precedence CLI → `ATLAS_CONTENT_LOOP_MIN_REPEATS` →
    /// built-in [`CONTENT_LOOP_MIN_REPEATS`].
    ///
    /// Was a `OnceLock` plus a `set_watchdog_params` installer. Two problems,
    /// both fixed by building the value where it is known and carrying it:
    /// the tunables are per-model, so a `OnceLock` kept the first model's
    /// table for every model after it; and the installer ran from
    /// `log_behavior_audit`, which serve calls *after* it spawns the
    /// scheduler thread — the reader's `unwrap_or(&DEFAULT)` was the only
    /// reason that ordering was survivable.
    pub fn from_behavior(
        b: &atlas_kernels::ModelBehavior,
        max_inter_tool_prose_cli: Option<u32>,
        content_loop_min_repeats_cli: Option<u32>,
    ) -> Self {
        // Install the per-model A4 reasoning floor alongside the watchdog
        // config (same boot-time, same behavior source).
        crate::scheduler::sample_step::set_min_reasoning_floor(b.min_reasoning_floor_tokens);
        let mut p = Self {
            think_loop_min_repeats: b.think_loop_min_repeats as usize,
            think_loop_scan_window: b.think_loop_scan_window as usize,
            confidence_early_stop: b.confidence_early_stop,
            confidence_run_length: b.confidence_run_length,
            fuzzy_repeat_tolerance_div: b.fuzzy_repeat_tolerance_div as usize,
            max_inter_tool_prose: b.max_inter_tool_prose,
            max_post_think_content_tokens: b.max_post_think_content_tokens,
            rollback_resteer: b.rollback_resteer,
            honor_eos_inside_thinking: b.honor_eos_inside_thinking,
            enable_think_loop_watchdog: b.enable_think_loop_watchdog,
            content_loop_min_repeats: None,
        };
        // P2-1 (2026-07-09): `max_inter_tool_prose` (384) was tuned as an
        // `<invoke>`-dormant-opener WANDER bound, but opencode arms tools on
        // every turn, so a legitimate PLAN / analysis prose turn
        // (`grammar_state.is_some()` ⇒ "tool turn") is subject to it and gets
        // guillotined mid-sentence (finish=length) — the "worst run ever"
        // 6-turn session died writing an API plan at 385 tokens. The REPEATING
        // wander is already caught by the content-loop + SimHash watchdogs
        // independently; this budget's residual job is only the non-repeating
        // dormant-opener burn.
        let env = match std::env::var("ATLAS_MAX_INTER_TOOL_PROSE") {
            Ok(v) => match v.parse::<u32>() {
                Ok(n) => Some(n),
                Err(_) => {
                    // A set-but-unparseable override is a config error, not
                    // an absent one — silently keeping the model default here
                    // is how a truncation "fix" fails to apply (#328 class).
                    tracing::warn!(
                        value = %v,
                        "ATLAS_MAX_INTER_TOOL_PROSE is set but not a u32; ignoring it"
                    );
                    None
                }
            },
            Err(_) => None,
        };
        p.max_inter_tool_prose =
            resolve_max_inter_tool_prose(p.max_inter_tool_prose, env, max_inter_tool_prose_cli);
        p.content_loop_min_repeats = content_loop_min_repeats_cli.or(parse_env_u32(
            "ATLAS_CONTENT_LOOP_MIN_REPEATS",
            std::env::var("ATLAS_CONTENT_LOOP_MIN_REPEATS")
                .ok()
                .as_deref(),
        ));
        p
    }

    /// The content-loop detector params in force for one sequence: the
    /// request's own `repetition_detection` object outranks the operator
    /// override; `None` = the built-in constants. Periods stay at the
    /// built-in range — only the repeat threshold is operator-tunable.
    pub fn content_loop_params(
        &self,
        request: Option<crate::api::inference_types::RepetitionDetectionParams>,
    ) -> Option<crate::api::inference_types::RepetitionDetectionParams> {
        request.or_else(|| {
            self.content_loop_min_repeats.map(|n| {
                crate::api::inference_types::RepetitionDetectionParams {
                    min_pattern_size: CONTENT_LOOP_PERIOD_MIN as u32,
                    max_pattern_size: CONTENT_LOOP_PERIOD_MAX as u32,
                    min_count: n,
                }
            })
        })
    }
}

/// Parse an optional numeric env override. A set-but-unparseable value is a
/// config error, not an absent one (#328 class) — warn, never silently drop.
fn parse_env_u32(name: &str, v: Option<&str>) -> Option<u32> {
    let v = v?;
    match v.trim().parse::<u32>() {
        Ok(n) => Some(n),
        Err(_) => {
            tracing::warn!(value = %v, "{name} is set but not a u32; ignoring it");
            None
        }
    }
}

/// Resolve whether the content-loop watchdog is armed for this run.
///
/// Precedence, highest wins: `--content-loop-watchdog` (CLI) →
/// `ATLAS_CONTENT_LOOP_WATCHDOG` (env, `1`/`true`/`0`/`false`) → MODEL.toml
/// `[behavior].enable_loop_watchdog`. Before this resolver the MODEL.toml
/// value was FINAL on the shipped image (MODEL.toml is baked in at build
/// time), so a model that opted in — e.g. the qwen3-next family, for its
/// run-on incident — dragged every derivative checkpoint's operators along
/// with no reachable off-switch short of `ATLAS_DISABLE_WATCHDOGS=1`
/// (which disarms EVERY guard, not this one).
pub fn resolve_content_loop_watchdog(toml: bool, env: Option<&str>, cli: Option<bool>) -> bool {
    if let Some(cli) = cli {
        return cli;
    }
    match env {
        None => toml,
        Some(v) => {
            let v = v.trim();
            if v == "1" || v.eq_ignore_ascii_case("true") {
                true
            } else if v == "0" || v.eq_ignore_ascii_case("false") {
                false
            } else {
                tracing::warn!(
                    value = %v,
                    "ATLAS_CONTENT_LOOP_WATCHDOG is set but not 1/true/0/false; \
                     keeping the MODEL.toml value"
                );
                toml
            }
        }
    }
}

/// Resolve the effective inter-tool prose budget (#328).
///
/// Precedence, highest wins: `--max-inter-tool-prose` (CLI, typed per
/// launch next to the model) → `ATLAS_MAX_INTER_TOOL_PROSE` (env) →
/// MODEL.toml `[behavior].max_inter_tool_prose` → the shared default
/// (`atlas_kernels::DEFAULT_MAX_INTER_TOOL_PROSE`, already folded into
/// `toml` by the build-time parse).
///
/// 0 means "guard disabled" and maps to `u32::MAX`: the check sites fire
/// on `prose_tokens > max`, so a literal 0 would end every tool-armed
/// response at its FIRST content token — the exact opposite of what an
/// operator writing 0 is asking for.
pub fn resolve_max_inter_tool_prose(toml: u32, env: Option<u32>, cli: Option<u32>) -> u32 {
    let v = cli.or(env).unwrap_or(toml);
    if v == 0 { u32::MAX } else { v }
}

// The three vocabulary masks that lived here as `OnceLock<Arc<[bool]>>` plus a
// `set_*` each are now `scheduler::vocab_masks::VocabMasks`, returned by
// `resolve_tokenizer_runtime` like every other tokenizer-derived value and
// carried down through `scheduler::run`.
//
// They were the sharpest hazard in the tree: each is INDEXED BY TOKEN ID, so a
// mask outliving the vocabulary that built it does not fail — it classifies the
// wrong ids, and the logit processors suppress the wrong tokens. Silently.

/// F2 (2026-04-26): cap on free-text tokens between successive
/// `<tool_call>` opens when `tool_choice="auto"`. The grammar FSM
/// in `auto` mode (grammar.rs:461-462) sets `at_least_one=false`
/// and `stop_after_first=false`, so `is_terminated()` stays false
/// forever after the first tool call — the model can emit
/// prose↔tool↔prose↔tool indefinitely. Counted across non-thinking,
/// non-tool-body tokens only.
///
/// Aliases the atlas-kernels default rather than restating it: P2-1
/// raised this constant to 3072 while the kernels-side default (the one
/// `from_behavior` actually reads for every model) stayed 384, so the
/// "fixed" budget kept amputating agent narration for a month (#328).
pub const MAX_INTER_TOOL_PROSE: u32 = atlas_kernels::DEFAULT_MAX_INTER_TOOL_PROSE;

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
pub fn detect_thinking_token_loop(tokens: &[u32], wp: WatchdogParams) -> bool {
    detect_thinking_token_loop_with(tokens, None, wp)
}

/// Per-sequence override variant of [`detect_thinking_token_loop`].
/// When `override_` is `Some(p)`, uses `p.min_pattern_size`,
/// `p.max_pattern_size`, `p.min_count` as the period and repeat
/// thresholds — exactly mirroring vLLM's `RepetitionDetectionParams`
/// (`sampling_params.py:111-144`). When `None`, falls back to the
/// run's `WatchdogParams` so existing callers without per-request
/// configuration are byte-identical to before.
pub fn detect_thinking_token_loop_with(
    tokens: &[u32],
    override_: Option<crate::api::inference_types::RepetitionDetectionParams>,
    wp: WatchdogParams,
) -> bool {
    let (period_min, period_max, min_repeats) = match override_ {
        Some(p) => (
            p.min_pattern_size as usize,
            p.max_pattern_size as usize,
            p.min_count as usize,
        ),
        None => (
            THINK_LOOP_PERIOD_MIN,
            THINK_LOOP_PERIOD_MAX,
            wp.think_loop_min_repeats,
        ),
    };
    let scan_window = match override_ {
        Some(_) => 0, // vLLM-anchored detector ignores scan_window
        None => wp.think_loop_scan_window,
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
    override_: Option<crate::api::inference_types::RepetitionDetectionParams>,
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
    override_: Option<crate::api::inference_types::RepetitionDetectionParams>,
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

#[cfg(test)]
mod inter_tool_prose_tests {
    use super::{WatchdogParams, resolve_max_inter_tool_prose};

    #[test]
    fn default_prose_budget_is_plan_friendly() {
        // Regression: 384 amputated legitimate plan turns (2026-07-09).
        const {
            assert!(
                super::MAX_INTER_TOOL_PROSE >= 2048,
                "inter-tool prose budget must fit a typical plan/analysis turn"
            );
        };
    }

    #[test]
    fn resolved_default_budget_is_plan_friendly() {
        // #328: the const assert above was green for a month while every
        // served model got 384 — production reads the KERNELS-side default
        // through `from_behavior`, not the constant. Assert the RESOLVED
        // value, i.e. what `handle_content_token` actually compares against.
        let p = WatchdogParams::from_behavior(&atlas_kernels::ModelBehavior::default(), None, None);
        assert!(
            p.max_inter_tool_prose >= 2048,
            "resolved inter-tool prose budget must fit a plan/analysis turn \
             (got {}) — the build-time and lib defaults have drifted",
            p.max_inter_tool_prose
        );
    }

    #[test]
    fn prose_budget_precedence_is_cli_env_toml() {
        // MODEL.toml value stands alone.
        assert_eq!(resolve_max_inter_tool_prose(384, None, None), 384);
        // Env outranks MODEL.toml (P2-1 contract, unchanged).
        assert_eq!(resolve_max_inter_tool_prose(384, Some(8192), None), 8192);
        // CLI outranks both — it is typed per launch next to the model.
        assert_eq!(
            resolve_max_inter_tool_prose(384, Some(8192), Some(4096)),
            4096
        );
        assert_eq!(resolve_max_inter_tool_prose(384, None, Some(4096)), 4096);
    }

    #[test]
    fn prose_budget_zero_disables_instead_of_instant_firing() {
        // The check sites fire on `prose_tokens > max`, so a literal 0 would
        // truncate every tool-armed response at its first content token.
        // 0 must mean "guard off" from every source.
        assert_eq!(resolve_max_inter_tool_prose(0, None, None), u32::MAX);
        assert_eq!(resolve_max_inter_tool_prose(384, Some(0), None), u32::MAX);
        assert_eq!(
            resolve_max_inter_tool_prose(384, Some(8192), Some(0)),
            u32::MAX
        );
    }
}

#[cfg(test)]
mod content_loop_override_tests {
    use super::{
        CONTENT_LOOP_MIN_REPEATS, CONTENT_LOOP_PERIOD_MAX, CONTENT_LOOP_PERIOD_MIN, WatchdogParams,
        detect_content_token_loop_with, resolve_content_loop_watchdog,
    };
    use crate::api::inference_types::RepetitionDetectionParams;

    #[test]
    fn watchdog_arming_precedence_is_cli_env_toml() {
        // MODEL.toml value stands alone (both polarities).
        assert!(resolve_content_loop_watchdog(true, None, None));
        assert!(!resolve_content_loop_watchdog(false, None, None));
        // Env outranks MODEL.toml — the shipped image bakes MODEL.toml in,
        // so without this rung an opted-in model has no reachable off-switch.
        assert!(!resolve_content_loop_watchdog(true, Some("0"), None));
        assert!(resolve_content_loop_watchdog(false, Some("true"), None));
        // CLI outranks both.
        assert!(!resolve_content_loop_watchdog(true, Some("1"), Some(false)));
        assert!(resolve_content_loop_watchdog(false, Some("0"), Some(true)));
        // Unparseable env keeps the MODEL.toml value (and warns at the site).
        assert!(resolve_content_loop_watchdog(true, Some("banana"), None));
    }

    #[test]
    fn min_repeats_cli_reaches_the_resolved_params() {
        let p =
            WatchdogParams::from_behavior(&atlas_kernels::ModelBehavior::default(), None, Some(5));
        assert_eq!(p.content_loop_min_repeats, Some(5));
        let eff = p.content_loop_params(None).expect("override present");
        assert_eq!(eff.min_count, 5);
        assert_eq!(eff.min_pattern_size as usize, CONTENT_LOOP_PERIOD_MIN);
        assert_eq!(eff.max_pattern_size as usize, CONTENT_LOOP_PERIOD_MAX);
    }

    #[test]
    fn request_params_outrank_the_operator_override() {
        let p = WatchdogParams {
            content_loop_min_repeats: Some(5),
            ..WatchdogParams::default()
        };
        let req = RepetitionDetectionParams {
            min_pattern_size: 4,
            max_pattern_size: 8,
            min_count: 2,
        };
        let eff = p.content_loop_params(Some(req)).expect("request present");
        assert_eq!(eff.min_count, 2);
        assert_eq!(eff.min_pattern_size, 4);
    }

    #[test]
    fn unset_override_keeps_the_historical_constants() {
        let p = WatchdogParams::from_behavior(&atlas_kernels::ModelBehavior::default(), None, None);
        assert_eq!(p.content_loop_min_repeats, None);
        assert!(p.content_loop_params(None).is_none());
    }

    #[test]
    fn raised_min_repeats_passes_code_shaped_period_2_tails() {
        // Daniel's firing shape (#328 follow-up): 48 content tokens ending in
        // a period-2 pattern repeated 3 times — closing-brace/newline-class
        // code tails. At the
        // built-in threshold (3) the detector fires; an operator raising the
        // threshold to 5 lets the same tail through while a genuine 5-repeat
        // attractor still trips.
        let mut toks: Vec<u32> = (0..48u32).collect();
        toks.extend_from_slice(&[7, 8, 7, 8, 7, 8]); // 3 end-anchored repeats
        assert!(detect_content_token_loop_with(&toks, None));
        const { assert!(CONTENT_LOOP_MIN_REPEATS == 3) };
        let relaxed = RepetitionDetectionParams {
            min_pattern_size: CONTENT_LOOP_PERIOD_MIN as u32,
            max_pattern_size: CONTENT_LOOP_PERIOD_MAX as u32,
            min_count: 5,
        };
        assert!(!detect_content_token_loop_with(&toks, Some(relaxed)));
        toks.extend_from_slice(&[7, 8, 7, 8]); // now 5 repeats
        assert!(detect_content_token_loop_with(&toks, Some(relaxed)));
    }
}

#[cfg(test)]
mod hard_limit_tests {
    //! DS4F hard-limit lane (2026-07-21): the three guards that stop generation
    //! running past its declared ceilings (R1X overrun: past both max_tokens
    //! and max_seq_len once a long `<think>` block engaged). Tested on the pure
    //! decision cores (no `ActiveSeq` fixture — mirrors `budget_tests` /
    //! `cc6_envelope_streak_tests`). The wiring (thinking tokens calling
    //! `consume_generation_budget`, the guards firing in the two decode paths)
    //! is exercised by the behavioral T1 repro.
    use super::{eos_suppressed_by_thinking, hard_ceiling_hit, seqlen_force_stop};

    #[test]
    fn seqlen_guard_disabled_when_unset() {
        // max_seq_len == 0 → never fires, whatever the position (no-op default
        // for un-inited paths / unit tests / the direct-mode baseline).
        assert!(!seqlen_force_stop(0, 0));
        assert!(!seqlen_force_stop(8191, 0));
        assert!(!seqlen_force_stop(usize::MAX, 0));
    }

    #[test]
    fn seqlen_guard_fires_one_token_before_ceiling() {
        // §C-3 / §E-3: `position + 1 >= max_seq_len`. At 8192 the sequence must
        // stop before writing KV at the ceiling, and never beyond.
        assert!(!seqlen_force_stop(8190, 8192), "room for one more token");
        assert!(seqlen_force_stop(8191, 8192), "next token would be at 8191");
        assert!(seqlen_force_stop(8192, 8192), "already at the ceiling");
        assert!(
            seqlen_force_stop(9000, 8192),
            "past the ceiling never continues"
        );
    }

    #[test]
    fn hard_ceiling_hit_on_budget_or_seqlen() {
        // §C-1/§C-2: exhausted completion budget OR context ceiling reached.
        assert!(
            hard_ceiling_hit(0, 10, 8192),
            "remaining==0 is a hard ceiling"
        );
        assert!(
            hard_ceiling_hit(500, 8191, 8192),
            "seq-len ceiling is a hard ceiling"
        );
        assert!(
            !hard_ceiling_hit(500, 10, 8192),
            "budget + room left → no ceiling"
        );
        assert!(
            !hard_ceiling_hit(500, 10, 0),
            "max_seq_len unset → only budget matters"
        );
    }

    #[test]
    fn eos_reachable_at_hard_ceiling_even_inside_thinking() {
        // §C-2: EOS is suppressed inside `<think>` UNTIL a hard ceiling — then a
        // model-sampled EOS must be honored so generation cannot overrun.
        assert!(
            eos_suppressed_by_thinking(true, false),
            "inside thinking, no ceiling → suppress (unchanged baseline)"
        );
        assert!(
            !eos_suppressed_by_thinking(true, true),
            "inside thinking, hard ceiling → EOS must fire (the fix)"
        );
        assert!(
            !eos_suppressed_by_thinking(false, false),
            "outside thinking → never suppressed"
        );
    }
}
