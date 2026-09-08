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
//! Correctness note, and its LIMIT. For the token emitted by one step, drafter
//! KV cannot corrupt output: the target verifies every draft, so a wrong or
//! missing row costs acceptance, not correctness. That argument covers one
//! step and does not extend to the RECURRENT state, because the accepted-draft
//! count selects between numerically distinct state paths (full accept keeps
//! the verify kernel's own state; a reject restores from the batched
//! intermediate; neither is bit-equal to an M=1 decode). A drafter fed by a
//! DIFFERENT request therefore moves acceptance, and acceptance moves the SSM
//! state every later token is decoded from. Validity below is consequently
//! about three things, not two: not wasting the lever, not reading another
//! sequence's hiddens, and not adopting another SESSION's rows at all.

use spark_runtime::gpu::DevicePtr;

/// Carry the drafter's KV across turns instead of rebuilding (or, before this
/// existed, skipping) it on every warm turn. **ON by default.**
///
/// Inseparable from the drafter prefill, which owns the hidden buffer this
/// path reads: the call site is nested inside `!mtp_prefill_hidden.is_null()`,
/// so carry alone is inert, and prefill without carry is a measured −927
/// ms/turn loss. [`crate::model::drafter_context`] resolves both together and
/// is the single source of truth for the policy and its kill switch.
/// Minimum MATCHED prefix (tokens) before the Marconi SSM snapshot skip is worth
/// taking. Below this, take the KV-only path instead.
///
/// # Why a floor exists at all
///
/// Restoring an SSM snapshot skips the target's prefill, so
/// `mtp_prefill_capture_len` stays 0, the `captured >= prompt_len` guard at
/// `speculative.rs:194` fails, `prefill_drafter` is skipped and the drafter
/// starts EMPTY (the defect this module's carry closes for SAME-SESSION turns —
/// but a fresh request matching only a shared chat-template preamble has no
/// previous turn to carry from, so carry cannot fire).
///
/// Measured at C=1, identical-prompt reps (full-prompt hits), warm reps vs
/// caching-off, 2026-07-28:
/// ```text
///    99 matched tokens   23.85 vs 26.4   -9.7%   LOSS
///   219 matched tokens   24.15 vs 22.0   +9.8%   WIN
///   349 matched tokens   23.55 vs 21.9   +7.5%
///   629 matched tokens   22.45 vs 20.3  +10.6%
/// ```
/// The crossover is SHARP, between ~99 and ~219. 256 sits inside the win region
/// and is block-aligned (16 x 16-token blocks). On preamble-only traffic the
/// penalty was -6.8% at C=1 and -9.2% at C=2, and inert by C=4.
///
/// `ATLAS_MARCONI_MIN_TOKENS=<n>` overrides; 0 restores the previous
/// always-restore behaviour.
pub fn marconi_min_tokens() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("ATLAS_MARCONI_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256)
    })
}

/// Is the carry ARMED, given how it is configured and whether MTP is
/// dispatching multiple sequences?
///
/// Pure, so the rule can be tested; `mtp_max_seqs()` caches its env read in a
/// `OnceLock` and a unit test cannot flip it. The env is read by the callers
/// below, at the boundary.
///
/// ★ CONFIGURED IS NOT ARMED, and conflating the two cost a night of GPU on
/// 2026-09-07. `ATLAS_MTP_MAX_SEQS` defaults to 32, so `multi_seq` is true on
/// an unconfigured serve and the carry is INERT no matter what
/// `DrafterContext` says. Anything that reports the carry's state to a human
/// must report THIS, not `cfg.carry`.
pub fn carry_armed_with(
    cfg: crate::model::drafter_context::DrafterContext,
    multi_seq: bool,
) -> bool {
    // Force-off in multi-seq MTP mode: the carry slot is single-sequence by
    // design (one slot, `active.len() == 1` assumption). See
    // `speculative::mtp_multi_seq_mode` for the contract.
    cfg.carry && !multi_seq
}

/// [`carry_armed_with`] against the live dispatch cap.
pub fn carry_armed(cfg: crate::model::drafter_context::DrafterContext) -> bool {
    carry_armed_with(cfg, crate::speculative::mtp_multi_seq_mode())
}

pub fn mtp_carry_drafter_enabled(levers: &crate::layers::ops::ModelLevers) -> bool {
    carry_armed(levers.drafter)
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
    /// The token sequence that produced these rows. `hidden_i` is a pure
    /// function of `tokens[0..=i]`, so the COMMON PREFIX with a later prompt
    /// bounds which rows that prompt may adopt — see [`Self::usable_by`],
    /// which truncates to that bound rather than demanding full equality.
    ///
    /// Prefix agreement is a bound, NOT an identity. Two unrelated requests
    /// rendered through one chat template agree on hundreds of tokens, so this
    /// field cannot answer "are these rows mine"; [`Self::session_hash`] does.
    pub tokens: Vec<u32>,
    /// The session that produced these rows, copied from
    /// `SequenceState::session_hash` at deposit.
    ///
    /// Without it the slot is a cross-request channel: it is MODEL-level (one
    /// slot for the whole engine, `types.rs`), it is filled by whichever
    /// sequence finished last, and prefix truncation alone admits any prompt
    /// sharing two leading tokens — which every templated request does.
    pub session_hash: u64,
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

    /// Do these rows belong to `session_hash`?
    ///
    /// The admission gate. Prefix agreement bounds WHICH rows are numerically
    /// reusable; this decides whether the entry may be reused AT ALL.
    ///
    /// ★ `0` REFUSES. A zero hash means the scheduler stamped no session, so
    /// there is nothing to verify ownership against. This deliberately differs
    /// from `SsmSnapshot::session_matches`, which treats 0 as "legacy tracking
    /// off, allow"; the closer precedent is the sibling single-slot in this
    /// same subsystem, whose `owns_capture` stamp requires a non-zero
    /// generation for the identical reason — blind beats poisoned.
    pub fn session_matches(&self, session_hash: u64) -> bool {
        session_hash != 0 && self.session_hash == session_hash
    }

    /// How much of this entry `prompt` may adopt.
    ///
    /// Refuses outright unless [`Self::session_matches`]; the prefix rules
    /// below only ever NARROW an entry this session already owns.
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
    pub fn usable_by(&self, prompt: &[u32], session_hash: u64) -> Option<(usize, usize)> {
        if !self.session_matches(session_hash) {
            return None;
        }
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
    /// The slot held another session's rows (or this request carries no
    /// session stamp). Distinct from `PrefixMismatch` on purpose: a prefix
    /// mismatch is a re-tokenized turn boundary and is expected, while this is
    /// the cross-request channel being refused, and reading one as the other
    /// is how it stayed open.
    ForeignSession {
        entry_session: u64,
        prompt_session: u64,
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
            CarryOutcome::ForeignSession {
                entry_session,
                prompt_session,
            } => write!(
                f,
                "foreign session (entry={entry_session:#x} prompt={prompt_session:#x})"
            ),
            CarryOutcome::NoHiddens => write!(f, "hidden store does not cover the append span"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The session every fixture below belongs to.
    const SESSION: u64 = 0x5E55_1014;
    /// A different live session, used as the intruder.
    const OTHER: u64 = 0x0DD0_0DD0;

    fn carried(tokens: &[u32], rows: usize, last_pair_key: Option<usize>) -> CarriedDrafter {
        CarriedDrafter {
            block_table: vec![1, 2, 3],
            rows,
            last_pair_key,
            tokens: tokens.to_vec(),
            session_hash: SESSION,
        }
    }

    /// ★ THE CROSS-REQUEST CHANNEL. A different session may not adopt the
    /// slot, no matter how much of the prompt agrees — here the entry's tokens
    /// are a FULL prefix of the intruder's prompt, which is the most
    /// permissive case the old prefix-only rule had.
    ///
    /// Before the session gate this returned `Some((4, 3))`.
    #[test]
    fn a_foreign_session_cannot_adopt_the_slot() {
        let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
        assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7], SESSION), Some((4, 3)));
        assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7], OTHER), None);
    }

    /// A request the scheduler stamped with no session cannot adopt either.
    /// Zero is "unknown", not "wildcard": there is nothing to verify ownership
    /// against, and blind beats poisoned.
    #[test]
    fn an_unstamped_request_cannot_adopt_the_slot() {
        let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
        assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7], 0), None);
        // And an entry deposited without a stamp is not adoptable by anyone,
        // including another unstamped request.
        let mut unstamped = carried(&[1, 2, 3, 4, 5], 4, Some(3));
        unstamped.session_hash = 0;
        assert_eq!(unstamped.usable_by(&[1, 2, 3, 4, 5, 6, 7], 0), None);
        assert_eq!(unstamped.usable_by(&[1, 2, 3, 4, 5, 6, 7], SESSION), None);
    }

    /// ★ THE REPORTED SHAPE, in the terms it was reported in: two unrelated
    /// requests rendered through ONE chat template share a long leading run of
    /// tokens, so the old `common >= 2` admission test passed on every pair.
    /// Prefix agreement is a bound on WHICH rows are reusable; it was never an
    /// identity, and this pins that it is no longer read as one.
    #[test]
    fn the_shared_template_prefix_is_not_an_identity() {
        // 64 tokens of identical system/tool preamble, then the two requests
        // diverge into their own user turns.
        let template: Vec<u32> = (0..64).collect();
        let mut mine = template.clone();
        mine.extend_from_slice(&[900, 901, 902]);
        let mut theirs = template.clone();
        theirs.extend_from_slice(&[700, 701, 702]);

        let c = CarriedDrafter {
            block_table: vec![1, 2, 3],
            rows: 60,
            last_pair_key: Some(59),
            tokens: mine.clone(),
            session_hash: SESSION,
        };
        // The prefix rule alone would have admitted the stranger: 64 tokens in
        // common is far more than the two it required.
        assert_eq!(c.common_prefix_len(&theirs), 64);
        assert_eq!(c.usable_by(&theirs, OTHER), None);
        // The same session's own next turn still adopts, and still gets the
        // full entry — the gate narrows nothing it should not.
        assert_eq!(c.usable_by(&mine, SESSION), Some((60, 59)));
    }

    /// ★ The arming rule, and the trap it exists to name: a serve can report
    /// `carry=ON` from its `DrafterContext` and still never carry, because the
    /// MTP dispatch cap defaults to 32 and force-disables it. Both callers of
    /// this rule — the runtime gate and the startup report — must agree, which
    /// is why there is exactly one function.
    #[test]
    fn configured_carry_is_not_armed_carry_under_a_multi_seq_cap() {
        use crate::model::drafter_context::DrafterContext;
        let both = DrafterContext {
            prefill: true,
            carry: true,
        };
        let prefill_only = DrafterContext {
            prefill: true,
            carry: false,
        };
        // Single-sequence dispatch: configured ON is armed.
        assert!(carry_armed_with(both, false));
        // The SHIPPED default (cap 32 => multi_seq): configured ON, inert.
        assert!(
            !carry_armed_with(both, true),
            "a >1 dispatch cap must force the carry off"
        );
        // Configured off is off either way.
        assert!(!carry_armed_with(prefill_only, false));
        assert!(!carry_armed_with(prefill_only, true));
    }

    /// The gate's truth table, stated once, since two call sites read it.
    #[test]
    fn session_matches_is_equality_and_refuses_zero() {
        let c = carried(&[1, 2, 3], 2, Some(1));
        assert!(c.session_matches(SESSION));
        assert!(!c.session_matches(OTHER));
        assert!(!c.session_matches(0), "zero is unknown, not wildcard");
    }

    /// A foreign-session refusal must not be reported as a prefix mismatch.
    /// Reading one as the other is precisely how the channel stayed invisible:
    /// `PrefixMismatch` is the expected, benign outcome of a re-tokenized turn
    /// boundary, so it draws no attention in a log.
    #[test]
    fn the_two_refusals_do_not_read_alike() {
        let foreign = CarryOutcome::ForeignSession {
            entry_session: SESSION,
            prompt_session: OTHER,
        }
        .to_string();
        let mismatch = CarryOutcome::PrefixMismatch {
            common: 4,
            entry_rows: 9,
        }
        .to_string();
        assert!(foreign.contains("foreign session"), "{foreign}");
        assert!(!foreign.contains("prefix mismatch"), "{foreign}");
        assert!(mismatch.contains("prefix mismatch"), "{mismatch}");
        assert_ne!(foreign, mismatch);
    }

    #[test]
    fn usable_by_keeps_everything_when_the_whole_entry_matches() {
        let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
        // pair key 3 consumed tokens[0..=4]; all 5 match.
        assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7], SESSION), Some((4, 3)));
    }

    #[test]
    fn usable_by_truncates_the_tail_instead_of_refusing() {
        // Divergence at index 4 => common = 4 => highest usable key is 2, so
        // one row is dropped. This is the chat-template re-tokenization case
        // that made full-match adoption refuse every warm turn.
        let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
        assert_eq!(c.usable_by(&[1, 2, 3, 4, 9, 6, 7], SESSION), Some((3, 2)));
        // Divergence at index 2 => common = 2 => only key 0 survives.
        assert_eq!(c.usable_by(&[1, 2, 9, 9], SESSION), Some((1, 0)));
    }

    #[test]
    fn usable_by_declines_when_nothing_survives() {
        let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
        // Fewer than 2 tokens in common: not even pair key 0 is usable.
        assert_eq!(c.usable_by(&[1, 9, 9], SESSION), None);
        assert_eq!(c.usable_by(&[], SESSION), None);
        // No rows, or no tracked key.
        assert_eq!(
            carried(&[1, 2, 3], 0, Some(1)).usable_by(&[1, 2, 3, 4], SESSION),
            None
        );
        assert_eq!(carried(&[1, 2, 3], 2, None).usable_by(&[1, 2, 3, 4], SESSION), None);
    }

    #[test]
    fn usable_by_never_drops_more_rows_than_exist() {
        // A compacted entry: 2 rows but a far-ahead key. Truncating to a low
        // common prefix must decline rather than underflow.
        let c = carried(&[1, 2, 3, 4, 5, 6], 2, Some(4));
        assert_eq!(c.usable_by(&[1, 2, 9], SESSION), None);
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
        // Exclusive end 198 still omits row 198; this is the adjacent boundary.
        assert_eq!(plan_append(97, 200, 97, 198), None);
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
        assert_eq!(merge_interval((10, 20), 0, 5), (0, 5));
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
