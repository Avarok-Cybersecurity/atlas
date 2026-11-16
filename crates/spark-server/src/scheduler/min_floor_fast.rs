// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the `min_tokens_gpu_argmax` row: a greedy decode step
//! whose sequences still have a `min_tokens` floor to reach stays on the
//! device argmax, instead of being sent to the host sampling pipeline for
//! the floor's EOS mask alone. Off by default on every target.
//!
//! # What the floor costs today
//!
//! `process_decode_logits` (`decode_logits_step.rs`) takes the device argmax
//! only when no sequence in the batch needs the host pipeline, and an active
//! `min_tokens` floor is one of the reasons it gives: the pipeline's
//! `MinTokensEosMask` stage sets the EOS logits to minus infinity until the
//! floor is reached, and the device argmax does not. So every step of a
//! request that sets `min_tokens` (a fixed-length benchmark sets it to
//! `max_tokens`) copies the whole `[n, vocab]` logits to the host, widens
//! them to FP32 and scans them once per sequence: 3.97 MB and eight 248,070
//! element passes per step at 8 streams on the Qwen3.5-4B.
//! `AVAROK_DECODE_TIMING=1` prints both parts.
//!
//! # Why the mask needs no host pass
//!
//! With temperature 0 and neutral penalties the pipeline emits the argmax of
//! the masked logits, and masking an entry that is not the maximum does not
//! move the maximum. So an argmax of the raw logits is an argmax of the masked
//! ones unless it is one of the masked ids. That is checked after the fact
//! ([`hits_floor`]): when a floored sequence's argmax is one of its EOS ids,
//! the step is redone on the host, exactly as the two-id think mask already
//! is. The model wants to stop early only rarely, so the fallback is rare,
//! and a step that falls back emits what it emits today.
//!
//! A floored request with any penalty set ([`penalties_neutral`] false) keeps
//! the host pipeline, as it does with the row off: there the pipeline does
//! more than mask.
//!
//! # What does change: the tie-break
//!
//! The mask argument holds for one argmax, and the two paths do not share
//! one. The host pipeline keeps the FIRST index that holds the maximum; the
//! device argmax (`argmax_bf16`) keeps, among tied ids, the one with the lower
//! `index mod 1024` (its reduction prefers the lower thread). Exact top ties
//! are not rare in BF16 logits: 178 of 13,406 greedy positions in the quality dumps of an earlier
//! GPU run on this card (1.3%). So with the row on a floored greedy run is NOT
//! bit-identical to the row off. It emits what the same request emits without
//! a floor, which takes the device argmax already, up to that request's first
//! EOS pick. A device argmax with the host's tie-break (compare indices on
//! equal values in the reduction) would make the row exact; that kernel does
//! not exist yet.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Is the `min_tokens_gpu_argmax` row on (`AVAROK_MIN_TOKENS_GPU_ARGMAX`)?
/// Read once.
pub(super) fn row() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(spark_model::layers::ops::port_rows::min_tokens_gpu_argmax::row)
}

/// Whether a sequence still has a `min_tokens` floor to reach, which is when
/// the host pipeline would mask its EOS ids.
pub(super) fn floor_active(min_tokens: usize, output_len: usize) -> bool {
    min_tokens > output_len
}

/// Whether a request's penalties are exactly neutral. Only then is the EOS
/// mask the one thing the host pipeline does to a floored greedy row, so only
/// then may the row keep it on the device (the same five comparisons
/// `process_decode_logits` makes before it excuses a think-ended row).
pub(super) fn penalties_neutral(
    repetition: f32,
    presence: f32,
    frequency: f32,
    lz: f32,
    dry_multiplier: f32,
) -> bool {
    repetition == 1.0 && presence == 0.0 && frequency == 0.0 && lz == 0.0 && dry_multiplier == 0.0
}

/// Whether an active floor still forces the host pipeline: always when
/// `fast_ok` is false (the row off, or a request the row does not cover),
/// never when it is true (the argmax is checked afterwards instead).
pub(super) fn floor_needs_host(fast_ok: bool, min_tokens: usize, output_len: usize) -> bool {
    !fast_ok && floor_active(min_tokens, output_len)
}

/// Whether the device argmax `tok` of a sequence is a token the floor's EOS
/// mask would have removed, so the step must be redone on the host.
pub(super) fn hits_floor(tok: u32, min_tokens: usize, output_len: usize, eos: &[u32]) -> bool {
    floor_active(min_tokens, output_len) && eos.contains(&tok)
}

/// Steps redone on the host because a floored sequence's argmax was an EOS id.
static FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Count one such step; returns the running total (logged at powers of two
/// by the caller, so a workload that hits the floor every step stays quiet).
pub(super) fn note_fallback() -> u64 {
    FALLBACKS.fetch_add(1, Ordering::Relaxed) + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_floor_is_active_until_the_output_reaches_it() {
        assert!(floor_active(256, 0));
        assert!(floor_active(256, 255));
        assert!(!floor_active(256, 256));
        assert!(!floor_active(0, 0), "no floor requested");
    }

    #[test]
    fn the_row_off_keeps_every_floored_step_on_the_host() {
        assert!(floor_needs_host(false, 256, 10));
        assert!(!floor_needs_host(false, 256, 256));
        assert!(!floor_needs_host(false, 0, 0));
    }

    #[test]
    fn the_row_on_never_asks_for_the_host_up_front() {
        for (min, len) in [(256, 0), (256, 255), (256, 256), (0, 0)] {
            assert!(!floor_needs_host(true, min, len), "min={min} len={len}");
        }
    }

    #[test]
    fn a_request_with_any_penalty_keeps_its_floor_on_the_host() {
        assert!(penalties_neutral(1.0, 0.0, 0.0, 0.0, 0.0));
        assert!(!penalties_neutral(1.05, 0.0, 0.0, 0.0, 0.0), "repetition");
        assert!(!penalties_neutral(1.0, 0.5, 0.0, 0.0, 0.0), "presence");
        assert!(!penalties_neutral(1.0, 0.0, 0.5, 0.0, 0.0), "frequency");
        assert!(!penalties_neutral(1.0, 0.0, 0.0, 0.3, 0.0), "lz");
        assert!(!penalties_neutral(1.0, 0.0, 0.0, 0.0, 0.8), "dry");
        // The caller folds this into the row: a penalized request under a
        // floor needs the host exactly as it does with the row off.
        let row_on = true;
        let fast_ok = row_on && penalties_neutral(1.05, 0.0, 0.0, 0.0, 0.0);
        assert!(floor_needs_host(fast_ok, 256, 10));
        let fast_ok = row_on && penalties_neutral(1.0, 0.0, 0.0, 0.0, 0.0);
        assert!(!floor_needs_host(fast_ok, 256, 10));
    }

    #[test]
    fn only_an_eos_argmax_under_an_active_floor_falls_back() {
        let eos = [248_044u32, 248_045];
        // The usual case: the argmax is an ordinary token, so masking EOS
        // would not have moved it.
        assert!(!hits_floor(17, 256, 10, &eos));
        // The model wants to stop before the floor: the host must pick the
        // best non-EOS token.
        assert!(hits_floor(248_044, 256, 10, &eos));
        assert!(hits_floor(248_045, 256, 255, &eos));
        // Floor reached, or none requested: EOS is a legal pick.
        assert!(!hits_floor(248_044, 256, 256, &eos));
        assert!(!hits_floor(248_044, 0, 0, &eos));
        // No EOS ids registered: nothing was masked.
        assert!(!hits_floor(248_044, 256, 10, &[]));
    }

    #[test]
    fn masking_a_non_maximum_does_not_move_the_argmax() {
        // The claim the row rests on, on the stage's own mask function.
        use crate::scheduler::logit_processors::min_tokens_eos::mask_eos_before_min;
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv { (i, x) } else { (bi, bv) }
                })
                .0
        };
        let raw = [0.25f32, 3.5, -1.0, 2.0, 3.0];
        // EOS is not the maximum: same pick with and without the mask.
        let mut masked = raw;
        mask_eos_before_min(&mut masked, &[4], 0, 8);
        assert_eq!(argmax(&masked), argmax(&raw));
        assert!(!hits_floor(argmax(&raw) as u32, 8, 0, &[4]));
        // EOS is the maximum: the mask moves the pick, and the check sees it.
        let mut masked = raw;
        mask_eos_before_min(&mut masked, &[1], 0, 8);
        assert_ne!(argmax(&masked), argmax(&raw));
        assert!(hits_floor(argmax(&raw) as u32, 8, 0, &[1]));
    }

    #[test]
    fn fallbacks_are_counted() {
        let a = note_fallback();
        let b = note_fallback();
        assert!(b > a);
    }
}
