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
