// SPDX-License-Identifier: AGPL-3.0-only
// Design after the TierKV pager of arczhi/llama-tierkv (kv-paging-design.md; MIT, Copyright (c) 2023-2026 The ggml authors): MTP draft contexts are kept out of paging (the draft cache must stay aligned with the target), taken from a written description of the design; no code copied or fetched.

//! Responsibility: the windowed MTP draft KV of the window desk (lane L36).
//!
//! The desk never pages the MTP drafter's own KV (its block table is separate from the
//! target's, and TierKV's experience is that paging draft contexts misaligns them with the
//! target). What it bounds is what the drafter READS: with the desk on, every MTP head attends
//! its sink rows plus the last `--kv-window-tokens` rows of its KV, through lane L11's
//! `MtpHead::set_kv_window` process default (`layers/mtp_head/window.rs`), set here before the
//! model is built. Drafts only steer speculation (the target verifies every token), so the bound
//! changes acceptance, never the output distribution. With the desk off nothing is set and the
//! drafter reads its whole KV, byte-identical to the path without this module.

/// The drafter window for a desk that is `on` with a recent window of `window_tokens`
/// positions: `None` (no bound, the identity) when the desk is off or the window is zero.
pub fn window_for(on: bool, window_tokens: u32) -> Option<u32> {
    if on && window_tokens > 0 {
        Some(window_tokens)
    } else {
        None
    }
}

/// Publish `window` as every MTP head's default bound. `None` publishes nothing, so the
/// default (no bound) and any head-level setting stay exactly as they are.
pub fn apply(window: Option<u32>) {
    let Some(w) = window else {
        return;
    };
    crate::layers::mtp_head::window::set_default_kv_window(Some(w));
    tracing::info!(
        "kv window desk: MTP heads attend their sink rows plus the last {w} draft KV rows"
    );
}
