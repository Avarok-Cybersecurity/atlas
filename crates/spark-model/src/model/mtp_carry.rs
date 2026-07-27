// SPDX-License-Identifier: AGPL-3.0-only

//! Carry the MTP drafter's KV across turns of a session (ON by default; see
//! [`crate::model::drafter_context`] for the switch and the coupling).
//!
//! # The defect this closes
//!
//! The drafter is prompt-prefilled only on a COLD turn. On a WARM turn the
//! target reuses a cached prefix, so `try_mtp_prefill_capture` never sees a
//! chunk starting at 0, `mtp_prefill_capture_len` stays 0, the propose-site
//! guard `captured >= prompt_len` fails, and `prefill_drafter` is SKIPPED.
//! Proposer state is per-request, so the drafter then starts EMPTY and gains
//! one row per decoded token: measured **142 drafter KV rows at sequence
//! position 10,098**, and **987 of 1007 scored MLPerf-edge samples are warm**.
//! Measured cost of that blindness: **+0.079 p1 / +0.089 p2_uncond**, about
//! **+10% accepted tokens per verify step**, de-confounded from SSM warm
//! restore (which is only +0.0070 p1 on its own).
//!
//! # Why NOT just re-run the whole-prompt drafter prefill on warm turns
//!
//! Measured on GB10 2026-07-21: `prefill_drafter` over 11,947 rows costs
//! **1136 ms**, of which the `fc` GEMM alone is 874 ms. Warm TTFT on the same
//! rig is 1134 ms, so a full warm-turn rebuild roughly DOUBLES TTFT to buy
//! ~10% of decode. On the scored workload (turns average ~71 output tokens,
//! ~3.7 s of generation) that trades ~370 ms of decode for ~1136 ms of TTFT —
//! a net wall-clock LOSS on the metric Atlas currently wins 1.80x. The two
//! per-row loops are only 7.6% of it, so batching them does not rescue it, and
//! `dense_gemm_tc` measured 21% SLOWER than the scalar kernel at this shape.
//!
//! # The mechanism
//!
//! A turn's prompt is a strict extension of the previous turn's full sequence
//! — that is exactly why the prefix cache hits. So the drafter rows the
//! previous turn already built ARE the rows this turn needs; only the tail is
//! missing. This module keeps the previous turn's drafter KV alive in a
//! single model-level slot (MTP is concurrency-1: every spec path is gated
//! `active.len() == 1`) and appends only the new span.
//!
//! Conventions, which is where this code kills people:
//!   * drafter row `r` holds pair key `k` = `(embed(t_{k+1}), hidden_k)`, RoPE
//!     `k + 1`. Rows are COMPACTED (dense slots) while RoPE stays in sequence
//!     space, so key gaps are already the norm — a partial append is safe.
//!   * `mtp_prefill_hidden` row `i` holds `hidden_i`. (The catch-up ring uses
//!     the OTHER convention — label `n` holds `hidden_{n-1}`. Do not mix them;
//!     that off-by-one was live until `d9984089`.)
//!
//! Correctness note: drafter KV can never corrupt output. The target verifies
//! every draft, so a wrong or missing drafter row costs acceptance, not
//! correctness. Validity below is therefore about not wasting the lever, and
//! about not reading another sequence's hiddens.

use spark_runtime::gpu::DevicePtr;

/// Carry the drafter's KV across turns instead of rebuilding (or, before this
/// existed, skipping) it on every warm turn. **ON by default.**
///
/// Inseparable from the drafter prefill, which owns the hidden buffer this
/// path reads: the call site is nested inside `!mtp_prefill_hidden.is_null()`,
/// so carry alone is inert, and prefill without carry is a measured −927
/// ms/turn loss. [`crate::model::drafter_context`] resolves both together and
/// is the single source of truth for the policy and its kill switch.
pub fn mtp_carry_drafter_enabled() -> bool {
    crate::model::drafter_context::config().carry
}

/// `ATLAS_MTP_CARRY_DEBUG=1` — one line per adopt/carry decision. Cheap (no
/// device reads, no syncs), but still off by default so timed legs stay quiet.
pub fn mtp_carry_debug() -> bool {
    std::env::var("ATLAS_MTP_CARRY_DEBUG").ok().as_deref() == Some("1")
}

/// The drafter KV of a finished turn, held for the next turn of the same
/// session. Single slot: MTP never runs at concurrency > 1, and one slot keeps
/// block ownership trivially safe (the blocks are owned here, or by a live
/// sequence, never both).
pub struct CarriedDrafter {
    /// Drafter KV blocks, moved out of the finished sequence's proposer state
    /// so `free_state` does not release them.
    pub block_table: Vec<u32>,
    /// Drafter rows resident in those blocks.
    pub rows: usize,
    /// Sequence-space pair key of the newest resident row.
    pub last_pair_key: Option<usize>,
    /// The token sequence that produced these rows. A later turn may adopt
    /// them only if its prompt starts with exactly these tokens — `hidden_i`
    /// is a pure function of `tokens[0..=i]`, so prefix equality is the whole
    /// validity condition.
    pub tokens: Vec<u32>,
}

impl CarriedDrafter {
    /// Length of the common prefix of `self.tokens` and `prompt`.
    pub fn common_prefix_len(&self, prompt: &[u32]) -> usize {
        self.tokens
            .iter()
            .zip(prompt.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// How much of this entry `prompt` may adopt.
    ///
    /// Pair key `k` consumed `tokens[0..=k + 1]`, so a key is usable exactly
    /// when the prompt agrees with those tokens. Requiring the WHOLE entry to
    /// match is too strict in practice: a chat template can re-tokenize the
    /// assistant/user boundary, so the tail of the previous turn's sequence
    /// need not reappear verbatim in the next turn's prompt (measured on the
    /// 27B rig — full-match adoption reported `prefix mismatch` on every warm
    /// turn). Truncating instead of refusing keeps the ~12k rows that DO
    /// match and loses only the handful that do not.
    ///
    /// Rows are append-only in increasing key order, so dropping `d` rows from
    /// the TAIL drops the `d` highest keys. `last_pair_key` is then clamped to
    /// `L - 2`, which can only OVERSTATE the surviving row's true key when the
    /// tail had gaps — and overstating merely starts the append later, i.e.
    /// costs coverage, never correctness. Rows beyond the returned count are
    /// overwritten by the append or never read (the drafter reads `seq_len`
    /// rows).
    ///
    /// Returns `(rows, last_pair_key)` to adopt, or `None` when nothing is
    /// usable.
    pub fn usable_by(&self, prompt: &[u32]) -> Option<(usize, usize)> {
        let k = self.last_pair_key?;
        if self.rows == 0 {
            return None;
        }
        let common = self.common_prefix_len(prompt);
        // Need at least tokens[0..=1] in common for pair key 0 to survive.
        let max_key = common.checked_sub(2)?;
        let key = k.min(max_key);
        let dropped = k - key;
        let rows = self.rows.checked_sub(dropped)?;
        if rows == 0 { None } else { Some((rows, key)) }
    }
}

/// Where a warm-turn append must start, given the carried state and the new
/// prompt, and where its hiddens must come from.
///
/// * `first_key` — the first pair key to write. `last_pair_key + 1` normally;
///   clamped up to `hidden_lo` when the hidden store does not reach back that
///   far. Skipping keys leaves no hole: rows are compacted, RoPE carries the
///   position, and a gap is already the steady-state shape of this row space.
/// * `rows` — how many pair keys get written: `first_key ..= prompt_len - 2`.
///
/// Returns `None` when there is nothing to append (the drafter already covers
/// the prompt) or when the hidden store cannot reach the first needed row.
pub fn plan_append(
    last_pair_key: usize,
    prompt_len: usize,
    hidden_lo: usize,
    hidden_hi: usize,
) -> Option<AppendPlan> {
    // Pair keys run 0 ..= prompt_len - 2 for a prompt of `prompt_len` tokens.
    let last_key_needed = prompt_len.checked_sub(2)?;
    let first_key = (last_pair_key + 1).max(hidden_lo);
    if first_key > last_key_needed {
        return None;
    }
    // Pair key k reads hidden row k, so the store must cover
    // [first_key, last_key_needed]; hidden_hi is exclusive.
    if hidden_hi <= last_key_needed || hidden_lo > first_key {
        return None;
    }
    Some(AppendPlan {
        first_key,
        rows: last_key_needed - first_key + 1,
    })
}

#[derive(Debug, PartialEq, Eq)]
pub struct AppendPlan {
    pub first_key: usize,
    pub rows: usize,
}

/// Byte offset of hidden row `pos` in a `[capacity, hidden_size]` BF16 store.
pub fn hidden_row_offset(base: DevicePtr, pos: usize, hidden_size: usize) -> DevicePtr {
    base.offset(pos * hidden_size * 2)
}

/// Merge a write of `[start, start + count)` into a single contiguous validity
/// interval `[lo, hi)`. Overlapping or abutting writes extend it; a disjoint
/// write REPLACES it, because one interval cannot describe two islands and
/// silently claiming the gap would hand the drafter another turn's hiddens.
pub fn merge_interval(cur: (usize, usize), start: usize, count: usize) -> (usize, usize) {
    let (lo, hi) = cur;
    let (ns, ne) = (start, start + count);
    if hi > lo && ns <= hi && ne >= lo {
        (lo.min(ns), hi.max(ne))
    } else {
        (ns, ne)
    }
}

/// Result of a carry attempt, for logging and tests.
#[derive(Debug, PartialEq, Eq)]
pub enum CarryOutcome {
    Adopted {
        rows: usize,
        appended: usize,
        first_key: usize,
    },
    NoCarry,
    PrefixMismatch {
        common: usize,
        entry_rows: usize,
    },
    NoHiddens,
}

impl std::fmt::Display for CarryOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CarryOutcome::Adopted {
                rows,
                appended,
                first_key,
            } => write!(
                f,
                "adopted rows={rows} appended={appended} first_key={first_key}"
            ),
            CarryOutcome::NoCarry => write!(f, "no carried state"),
            CarryOutcome::PrefixMismatch { common, entry_rows } => {
                write!(
                    f,
                    "prefix mismatch (common={common} entry_rows={entry_rows})"
                )
            }
            CarryOutcome::NoHiddens => write!(f, "hidden store does not cover the append span"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn carried(tokens: &[u32], rows: usize, last_pair_key: Option<usize>) -> CarriedDrafter {
        CarriedDrafter {
            block_table: vec![1, 2, 3],
            rows,
            last_pair_key,
            tokens: tokens.to_vec(),
        }
    }

    #[test]
    fn usable_by_keeps_everything_when_the_whole_entry_matches() {
        let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
        // pair key 3 consumed tokens[0..=4]; all 5 match.
        assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7]), Some((4, 3)));
    }

    #[test]
    fn usable_by_truncates_the_tail_instead_of_refusing() {
        // Divergence at index 4 => common = 4 => highest usable key is 2, so
        // one row is dropped. This is the chat-template re-tokenization case
        // that made full-match adoption refuse every warm turn.
        let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
        assert_eq!(c.usable_by(&[1, 2, 3, 4, 9, 6, 7]), Some((3, 2)));
        // Divergence at index 2 => common = 2 => only key 0 survives.
        assert_eq!(c.usable_by(&[1, 2, 9, 9]), Some((1, 0)));
    }

    #[test]
    fn usable_by_declines_when_nothing_survives() {
        let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
        // Fewer than 2 tokens in common: not even pair key 0 is usable.
        assert_eq!(c.usable_by(&[1, 9, 9]), None);
        assert_eq!(c.usable_by(&[]), None);
        // No rows, or no tracked key.
        assert_eq!(
            carried(&[1, 2, 3], 0, Some(1)).usable_by(&[1, 2, 3, 4]),
            None
        );
        assert_eq!(carried(&[1, 2, 3], 2, None).usable_by(&[1, 2, 3, 4]), None);
    }

    #[test]
    fn usable_by_never_drops_more_rows_than_exist() {
        // A compacted entry: 2 rows but a far-ahead key. Truncating to a low
        // common prefix must decline rather than underflow.
        let c = carried(&[1, 2, 3, 4, 5, 6], 2, Some(4));
        assert_eq!(c.usable_by(&[1, 2, 9]), None);
    }

    #[test]
    fn append_plan_covers_exactly_the_missing_pair_keys() {
        // Drafter holds keys 0..=97; prompt has 200 tokens => keys 0..=198.
        // Hidden store covers [97, 200).
        let p = plan_append(97, 200, 97, 200).unwrap();
        assert_eq!(
            p,
            AppendPlan {
                first_key: 98,
                rows: 101
            }
        );
    }

    #[test]
    fn append_plan_clamps_up_to_the_hidden_store_floor() {
        // Store only reaches back to 150, so keys 98..149 are unreachable.
        // Skipping them is safe: rows are compacted and RoPE carries position.
        let p = plan_append(97, 200, 150, 200).unwrap();
        assert_eq!(
            p,
            AppendPlan {
                first_key: 150,
                rows: 49
            }
        );
    }

    #[test]
    fn append_plan_declines_when_the_store_stops_short_of_the_last_key() {
        // Needs hidden row 198; store ends at 190 (exclusive).
        assert_eq!(plan_append(97, 200, 97, 190), None);
    }

    #[test]
    fn append_plan_declines_when_nothing_is_missing() {
        assert_eq!(plan_append(198, 200, 0, 200), None);
        assert_eq!(plan_append(250, 200, 0, 200), None);
    }

    #[test]
    fn append_plan_declines_on_a_degenerate_prompt() {
        assert_eq!(plan_append(0, 1, 0, 8), None);
        assert_eq!(plan_append(0, 0, 0, 8), None);
    }

    #[test]
    fn merge_interval_extends_on_overlap_and_abut() {
        assert_eq!(merge_interval((10, 20), 20, 5), (10, 25)); // abut
        assert_eq!(merge_interval((10, 20), 15, 10), (10, 25)); // overlap
        assert_eq!(merge_interval((10, 20), 5, 6), (5, 20)); // overlap below
    }

    #[test]
    fn merge_interval_replaces_on_a_gap() {
        // A disjoint write must NOT claim the gap: rows in it were never
        // written for this sequence.
        assert_eq!(merge_interval((10, 20), 30, 5), (30, 35));
        assert_eq!(merge_interval((0, 0), 30, 5), (30, 35));
    }

    #[test]
    fn common_prefix_len_is_the_validity_primitive() {
        let c = carried(&[1, 2, 3, 4], 3, Some(2));
        assert_eq!(c.common_prefix_len(&[1, 2, 3, 4, 5]), 4);
        assert_eq!(c.common_prefix_len(&[1, 2, 9, 4, 5]), 2);
        assert_eq!(c.common_prefix_len(&[]), 0);
    }
}
