// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-turn carry of the WHOLE DFlash proposer state.
//!
//! # The defect this closes
//!
//! A warm turn (prefix-cache + Marconi hit) skips the target prefill for the
//! cached prefix, so the DFlash 5-layer hidden capture only covers the
//! REPLAYED window. A fresh `DflashProposerState` therefore holds zeros for
//! every position before the replay start — yet `update_dflash_ctx_len_after_
//! prefill` sets `ctx_len` to the full prompt, so the drafter (a) conditions
//! on thousands of zero rows (measured: warm turns open 0/7, 0/7, 0/7
//! accepts before the real rows dominate) and (b) spends its first-propose
//! full ctx-KV precompute on those zeros (~0.41 s of the ~1.0 s warm TTFT at
//! 2.4K ctx).
//!
//! # Why WHOLE-state carry (vs the MTP blocks-only carry)
//!
//! The MTP carry moves `(block_table, rows, pair_key)` because MTP's drafter
//! context IS its KV. The DFlash drafter's context is richer: the
//! `ctx_hidden_acc` accumulator (5 captured layers per position — needed for
//! any KV re-precompute after a rewind), the paged ctx KV + its
//! `ctx_committed` watermark, and the per-slot stamped `ctx_positions`.
//! Moving the `Box<dyn ProposerState>` keeps all of it, and the adopt trims
//! it to the verified common token prefix.
//!
//! # Ownership
//!
//! One slot, same discipline as `mtp_carry`: the carried state is owned by
//! the slot XOR by a live sequence, never both. The store replaces (and
//! frees) any previous occupant; the adopt always TAKES the entry — on a
//! prefix mismatch or a coverage gap it frees the state rather than putting
//! it back, so a cold request self-clears the slot and its pool blocks
//! return before that request's own drafter allocation needs them.
//!
//! # Validity
//!
//! `entry.tokens` is the finished sequence's full token list (prompt +
//! completion). The adopt trims the carried ctx to
//! `common_prefix_len(entry.tokens, new_prompt)` — `ctx_hidden_acc` row `i`
//! is a pure function of `tokens[0..=i]`, so prefix equality is the whole
//! condition (the same argument as `mtp_carry::CarriedDrafter`). Contiguity
//! with this turn's capture requires `trim >= proc_start` (the first
//! position this prefill will process): below that the drafter would have a
//! hole of never-captured rows, which is exactly the zero-conditioning
//! defect again — the adopt bails instead.

use crate::speculative::ProposerState;

/// The finished turn's whole proposer state, held for the next turn.
pub struct CarriedDflashState {
    pub state: Box<dyn ProposerState>,
    /// Full token list (prompt + completion) of the sequence that produced
    /// the state. Adopt validity = prefix equality against the new prompt.
    pub tokens: Vec<u32>,
}

impl CarriedDflashState {
    pub fn common_prefix_len(&self, prompt: &[u32]) -> usize {
        self.tokens
            .iter()
            .zip(prompt.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }
}

/// Kill switch: presence of `ATLAS_NO_DFLASH_CARRY` disables the DFlash
/// whole-state carry (house convention — `=0` is NOT off). The shared
/// drafter-context policy (`ATLAS_NO_MTP_DRAFTER_CONTEXT`) is honored by the
/// call sites through `levers.drafter.carry`, same as the MTP carry.
pub fn dflash_carry_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_DFLASH_CARRY").is_none())
}

/// Floor on the carried ctx worth storing. A tiny context saves nothing and
/// the slot swap has real (if small) cost; aligned with the Marconi restore
/// floor so the two warm-turn mechanisms engage together.
pub fn dflash_carry_min_ctx() -> usize {
    crate::model::mtp_carry::marconi_min_tokens()
}
