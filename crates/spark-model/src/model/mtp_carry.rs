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
//! single model-level slot and appends only the new span.
//!
//! ★ WHY ONE SLOT IS SAFE, correctly stated. This used to read "MTP is
//! concurrency-1: every spec path is gated `active.len() == 1`". That is FALSE
//! and has been since the ladder campaign: the scheduler dispatches MTP
//! whenever `active.len() <= mtp_max_seqs()`, and that cap defaults to 32
//! (`speculative/ladder.rs`; `scheduler/phase_continue_prefills/spec_mixing.rs`
//! documents the same staleness). Only the n-gram and self-speculative lanes
//! still require `active.len() == 1`.
//!
//! What actually makes one slot safe is narrower and is enforced:
//! [`carry_armed_with`] force-disables the carry whenever the dispatch cap is
//! above 1, so the slot is only ever live in single-sequence mode. Even there,
//! ADMISSION is not gated by that cap — several sequences can be admitted and
//! prefill concurrently — which is why the shared hidden interval carries an
//! ownership stamp ([`StoreRange`]) rather than relying on a concurrency claim.
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
    *MARCONI_MIN.get_or_init(|| {
        std::env::var("ATLAS_MARCONI_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MARCONI_MIN_TOKENS)
    })
}

/// The shipped threshold, and the only place it is written down.
pub const DEFAULT_MARCONI_MIN_TOKENS: usize = 256;

static MARCONI_MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Pin the restore threshold from `--marconi-min-tokens`, before anything
/// reads it.
///
/// ★ WHY A SETTER AND NOT JUST THE ENV VAR. A KAT gate needs this value to be
/// part of its RECORD, and only recipe keys reach a record — an env var cannot,
/// so a run configured by `ATLAS_MARCONI_MIN_TOKENS` could not state that it
/// had been. Issue #936 measured a sharded BFCL draw disagreeing with the same
/// draw run whole on 12 of 995 samples through cross-request SSM snapshot
/// reuse; setting this high closes the consumer side and takes that to 2. A
/// configuration that fixes a correctness property is worthless if a record
/// cannot say it was used.
///
/// First writer wins, and a later call is IGNORED rather than panicking: the
/// value is a process-wide constant once anything has read it, and a serve
/// startup that set it twice would otherwise abort a running server over a
/// duplicate flag. Returns whether this call is the one that set it, so the
/// caller can warn if it lost the race — which means something read the
/// threshold before serve configured it, and the flag silently did nothing.
pub fn set_marconi_min_tokens(v: usize) -> bool {
    MARCONI_MIN.set(v).is_ok()
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
    // design. NOT because MTP is concurrency-1 — it is not, the cap defaults
    // to 32 — but because THIS check is what keeps the slot out of multi-seq
    // mode in the first place. See `speculative::mtp_multi_seq_mode`.
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
/// session. Single slot: the carry is force-disabled outside single-sequence
/// dispatch (see [`carry_armed_with`] — NOT because "MTP never runs at
/// concurrency > 1", which is false; the cap defaults to 32), and one slot
/// keeps block ownership trivially safe (the blocks are owned here, or by a
/// live sequence, never both).
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

/// The hidden-row interval, and WHOSE rows they are.
///
/// `mtp_prefill_hidden` is one model-level buffer indexed by ABSOLUTE sequence
/// position, with no per-sequence dimension, so an interval alone cannot say
/// who wrote the rows it covers. `gen` is that missing half.
///
/// ★ `gen == 0` NEVER MATCHES. It is the state of a range nothing has claimed,
/// and of every `SequenceState` built outside `alloc_sequence` (the mock and
/// test fakes, which draw no ticket). Treating it as a wildcard would reopen
/// the hole for exactly those constructors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreRange {
    /// The sequence generation that wrote these rows. 0 = unclaimed.
    /// Named `owner` because `gen` is a reserved keyword in edition 2024.
    pub owner: u64,
    /// First absolute position written.
    pub lo: usize,
    /// One past the last absolute position written.
    pub hi: usize,
}

impl StoreRange {
    /// Nothing claimed.
    pub const EMPTY: Self = Self {
        owner: 0,
        lo: 0,
        hi: 0,
    };

    /// The interval `reader_gen` may read, or `(0, 0)` when these rows belong
    /// to someone else.
    ///
    /// `(0, 0)` rather than an error because `plan_append` already refuses an
    /// interval that does not cover the span it needs — an empty interval
    /// covers nothing, so a foreign range degrades to the existing
    /// `CarryOutcome::NoHiddens` with no new control flow.
    pub fn visible_to(self, reader_gen: u64) -> (usize, usize) {
        if reader_gen != 0 && self.owner == reader_gen {
            (self.lo, self.hi)
        } else {
            (0, 0)
        }
    }
}

/// Merge a write into the interval, or take it over.
///
/// Same owner: [`merge_interval`], unchanged. Different owner (or an unclaimed
/// range): the writer REPLACES the interval with its own write and stamps it.
///
/// ★ REPLACE IS THE WHOLE POINT. Without it, `merge_interval` extends across
/// owners whenever the new chunk starts below the other sequence's high-water
/// mark, producing one interval whose low half is another request's rows —
/// which the `alloc_sequence` reset cannot catch, because the reset happened
/// before that other sequence wrote.
///
/// Coupled to `merge_interval`'s replace-on-a-gap behaviour for the SAME-owner
/// case: if that is ever relaxed to span a gap, this inherits the defect. It is
/// pinned by `merge_interval_replaces_on_a_gap`.
pub fn stamped_merge(cur: StoreRange, writer_gen: u64, start: usize, count: usize) -> StoreRange {
    if cur.owner != 0 && cur.owner == writer_gen {
        let (lo, hi) = merge_interval((cur.lo, cur.hi), start, count);
        StoreRange {
            owner: writer_gen,
            lo,
            hi,
        }
    } else {
        StoreRange {
            owner: writer_gen,
            lo: start,
            hi: start + count,
        }
    }
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
    /// The hidden rows in the store belong to another sequence. Reported for
    /// the debug log only — control flow degrades to `NoHiddens`, since an
    /// invisible interval covers nothing.
    ForeignHiddens {
        /// The generation that owns the rows.
        owner: u64,
        /// The generation that wanted to read them.
        expected: u64,
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
            CarryOutcome::ForeignHiddens { owner, expected } => write!(
                f,
                "hidden rows belong to sequence gen {owner}, not {expected}"
            ),
            CarryOutcome::NoHiddens => write!(f, "hidden store does not cover the append span"),
        }
    }
}

#[cfg(test)]
#[path = "mtp_carry_tests.rs"]
mod tests;
