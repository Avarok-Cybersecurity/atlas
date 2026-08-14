// SPDX-License-Identifier: AGPL-3.0-only
//
// `[behavior]` defaults that MUST be identical at build time and at run
// time. This file is a plain-`mod` of `lib.rs` AND `include!`d by the
// build script (`build_parse_behavior.rs`) — the build script cannot
// import the library it is building, and keeping two literals in sync by
// hand is exactly how #328 shipped: P2-1 (2026-07-09) raised the
// spark-server default to 3072 while the build-time parse default stayed
// 384, so every model without an explicit MODEL.toml pin kept truncating
// agentic prose at 384 tokens for another month. One literal, two
// consumers, no drift. Doc comments here use `//` only: `//!` is illegal
// at an `include!` site.

/// Default cap on free-text tokens between successive tool calls on a
/// tool-armed request (`[behavior].max_inter_tool_prose`).
///
/// 3072 is the P2-1 value (2026-07-09): 384 was tuned as an
/// `<invoke>`-dormant-opener wander bound, but agent frontends arm tools
/// on every turn, so a legitimate plan/analysis turn was guillotined
/// mid-sentence (#328: Pi.dev, opencode). Repeating wander is caught by
/// the content-loop + SimHash watchdogs independently; this budget's
/// residual job is the non-repeating dormant-opener burn, for which 3072
/// still sits well below a typical `max_tokens`. A model with measured
/// evidence for a tighter bound pins it in MODEL.toml (see
/// kernels/strix/qwen3.6-35b-a3b, 2026-06-10). 0 is reserved by the
/// runtime resolver to mean "guard disabled".
pub const DEFAULT_MAX_INTER_TOOL_PROSE: u32 = 3072;

/// Default `[behavior].max_thinking_budget` — the effort-ladder anchor E and
/// the budget for budgetless thinking-on requests. 256 is the historical
/// built-in every model inherited before MODEL.toml could override it.
/// Lifted here (2026-08-14, effort-ladder work) because it was three
/// hand-synced literals (lib.rs default, build-parse default, build-parse
/// `unwrap_or`) — the exact drift shape that shipped #328's 384-vs-3072 bug.
pub const DEFAULT_MAX_THINKING_BUDGET: u32 = 256;

/// Default `[behavior].effort_capped_at_ceiling` — whether qualitative
/// `reasoning_effort` levels are clamped at the model's effective ceiling E
/// (`max_thinking_budget` / `--max-thinking-budget`).
///
/// `false` preserves the historical ladder shape: high = 2E and xhigh = 4E
/// EXCEED the ceiling, exactly as the pre-symbolic absolutes did (512/1024
/// over the built-in 256). Parity at defaults is pinned by
/// `effort_ladder_at_default_ceiling_matches_the_historical_absolutes`.
///
/// `true` is for models with MEASURED non-monotonic degradation above their
/// ceiling — where a bigger thinking budget scores WORSE, so a client's
/// boilerplate `reasoning_effort: high` must not double a deliberately small
/// E (e.g. Qwen3.5-397B NVFP4, 2026-05-07 sweep: budget 256 is worse than
/// 128). The clamp binds ONLY the server-policy effort ladder; an explicit
/// client token budget (`thinking_token_budget` etc.) is never touched by it.
pub const DEFAULT_EFFORT_CAPPED_AT_CEILING: bool = false;
