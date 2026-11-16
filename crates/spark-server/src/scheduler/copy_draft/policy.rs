// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0. Source: src/models/qwen3_5/program/speculative/prompt_lookup.h:13-35 (propose, observe, confident: the lookup-round rule). Changes: translated from C++ to Rust; the proposal is observed lazily against the committed stream, by Atlas contributors.
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license). Applies to the source cursor and the copy-versus-MTP plan choice below.

//! Responsibility: the per-sequence copy planner and its selection table.
//!
//! [`CopyState`] holds one sequence's [`CopyIndex`], estimators and source
//! cursor. Each round [`plan`]:
//!
//! 1. syncs the index with the committed stream and scores what was in flight
//!    since the last plan against the tokens actually committed (no draft or
//!    rejected token ever enters the index: only `seq.tokens` and
//!    `last_token` are read, after the verdict);
//! 2. proposes a copy continuation: from the source cursor while the
//!    committed tokens keep following that source (KVMem's cursor, kept across
//!    fully accepted rounds and cleared on the first rejection), else from a
//!    fresh [`CopyIndex::longest`] lookup;
//! 3. chooses the round's plan with [`select`], a pure clause table.
//!
//! The proposal is scored even when the MTP chain is verified instead, the
//! way Cinference scores every lookup proposal: the target's committed tokens
//! are what verification would have accepted, so the window estimates warm up
//! without risking a round.

use super::config::{Config, Mode};
use super::estimator::{GateVerdict, WindowEstimates, YieldGate, common_prefix};
use super::index::{CopyIndex, MATCH_LIMIT, SyncOutcome};
use super::metrics::{METRICS, add};

/// Shortest lookup round (Cinference: a confident chain of at least 6).
pub(crate) const LOOKUP_MIN: usize = 6;
/// Lookup round lengths, so the variable-width verify captures a few graph
/// shapes instead of one per length.
pub(crate) const LOOKUP_BUCKETS: [usize; 4] = [6, 8, 12, 15];

/// What a verified round carried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Mtp,
    Chain,
    Lookup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cursor {
    /// `tokens()[next]` is the source's prediction for the next committed token.
    next: usize,
    /// Tokens the source agrees on so far (capped at [`MATCH_LIMIT`]).
    n: usize,
    window: usize,
}

struct Pending {
    kind: Option<Kind>,
    window: usize,
    tokens: Vec<u32>,
    at: usize,
}

/// A copy proposal for this round.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Proposal {
    pub window: usize,
    pub match_len: usize,
    pub confident: bool,
    pub tokens: Vec<u32>,
}

/// One sequence's copy drafter state.
#[derive(Default)]
pub(crate) struct CopyState {
    index: CopyIndex,
    est: WindowEstimates,
    gate: YieldGate,
    cursor: Option<Cursor>,
    /// The last proposal, scored at the next sync (estimator).
    shadow: Option<Pending>,
    /// What the last round actually verified, scored at the next sync (gate).
    inflight: Option<Pending>,
}

impl CopyState {
    pub fn with_index(index: CopyIndex) -> Self {
        Self {
            index,
            ..Self::default()
        }
    }

    pub fn index(&self) -> &CopyIndex {
        &self.index
    }

    pub fn estimates(&self) -> &WindowEstimates {
        &self.est
    }

    pub fn gate_state(&self) -> &YieldGate {
        &self.gate
    }

    pub fn has_cursor(&self) -> bool {
        self.cursor.is_some()
    }

    /// Reconcile with the committed stream and score what was pending.
    pub fn sync(&mut self, head: &[u32], last: Option<u32>) {
        let from = match self.index.sync(head, last) {
            SyncOutcome::Rebuilt => {
                self.cursor = None;
                self.shadow = None;
                self.inflight = None;
                return;
            }
            SyncOutcome::Appended { from } => from,
        };
        let toks = self.index.tokens();
        if from == toks.len() {
            return;
        }
        if let Some(mut c) = self.cursor {
            let mut alive = true;
            for &t in &toks[from..] {
                if c.next < toks.len() && toks[c.next] == t {
                    c.next += 1;
                    c.n = (c.n + 1).min(MATCH_LIMIT);
                } else {
                    alive = false;
                    break;
                }
            }
            self.cursor = alive.then_some(c);
        }
        if let Some(s) = self.shadow.take()
            && s.at <= toks.len()
        {
            self.est.observe(s.window, &s.tokens, &toks[s.at..]);
        }
        if let Some(f) = self.inflight.take()
            && f.at <= toks.len()
        {
            let accepted = common_prefix(&f.tokens, &toks[f.at..]);
            match f.kind {
                Some(Kind::Mtp) | None => self.gate.observe_mtp(accepted),
                Some(Kind::Chain) | Some(Kind::Lookup) => {
                    self.gate.observe_copy(accepted, f.tokens.len());
                    add(&METRICS.accepted, accepted);
                }
            }
        }
    }

    /// A copy continuation of `want` tokens, or `None` without a source.
    pub fn propose(&mut self, want: usize) -> Option<Proposal> {
        let len = self.index.len();
        self.shadow = None;
        if want == 0 || len < 2 {
            return None;
        }
        let c = match self.cursor {
            Some(c) if c.next < len => c,
            _ => {
                let m = self.index.longest();
                self.cursor = m.map(|m| Cursor {
                    next: m.at(),
                    n: m.n,
                    window: m.window,
                });
                self.cursor?
            }
        };
        let mut tokens = Vec::with_capacity(want);
        self.index.continuation(c.next, want, &mut tokens);
        self.shadow = Some(Pending {
            kind: None,
            window: c.window,
            tokens: tokens.clone(),
            at: len,
        });
        Some(Proposal {
            window: c.window,
            match_len: c.n,
            confident: self.est.confident(c.window, c.n),
            tokens,
        })
    }

    /// The yield gate's verdict for a chain of `width` tokens.
    pub fn gate(&mut self, width: usize, advantage: f32) -> GateVerdict {
        self.gate.decide(width, advantage)
    }

    /// Record what this round verifies (scored at the next sync).
    pub fn dispatch(&mut self, kind: Kind, tokens: &[u32]) {
        match kind {
            Kind::Chain => {
                add(&METRICS.chain_rounds, 1);
                add(&METRICS.proposed, tokens.len());
            }
            Kind::Lookup => {
                add(&METRICS.lookup_rounds, 1);
                add(&METRICS.proposed, tokens.len());
            }
            Kind::Mtp => {}
        }
        self.inflight = Some(Pending {
            kind: Some(kind),
            window: 0,
            tokens: tokens.to_vec(),
            at: self.index.len(),
        });
    }
}

/// The facts [`select`] decides on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Inputs {
    pub mode: Mode,
    /// The request asked for no speculative step (`seq.port.no_drafts`).
    pub no_drafts: bool,
    /// A grammar constrains this sequence.
    pub grammar: bool,
    /// MTP drafts pending for this round (the chain width a copy may fill).
    pub mtp_drafts: usize,
    /// Single sequence, no logprobs, not EP, MTP drafter (not DFlash).
    pub lookup_ok: bool,
    /// Lookup round length the slot and budget allow ([`lookup_len`]).
    pub lookup_len: usize,
    /// The copy source's agreement, when there is a source.
    pub match_len: Option<usize>,
    /// The window estimate says the lookup-round threshold holds.
    pub confident: bool,
    pub min_match: usize,
}

/// The round's choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Choice {
    /// No speculative step for this sequence.
    NoSpec,
    /// Verify the MTP drafter's chain as today.
    Mtp,
    /// Replace the MTP chain with the copy chain (same width).
    Chain,
    /// Verify the copy alone through the variable-width verify.
    Lookup,
}

/// The plan selection table, clause by clause (first match wins):
///
/// 1. the request asked for no drafts: no speculative step;
/// 2. the drafter is off: MTP;
/// 3. a grammar constrains the sequence: MTP (copies never cross a mask);
/// 4. no copy source: MTP;
/// 5. lookup rounds allowed, the sequence may take one, the estimate is
///    confident and the round would be at least [`LOOKUP_MIN`] long: lookup;
/// 6. chains allowed, MTP drafted at least one token and the source agrees on
///    at least `min_match` tokens: ask the yield gate, chain on take or probe;
/// 7. MTP.
///
/// `gate` runs only when clause 6 is reached (it moves the refusal streak).
pub(crate) fn select(i: &Inputs, gate: impl FnOnce() -> GateVerdict) -> Choice {
    if i.no_drafts {
        return Choice::NoSpec;
    }
    if i.mode == Mode::Off || i.grammar {
        return Choice::Mtp;
    }
    let Some(match_len) = i.match_len else {
        return Choice::Mtp;
    };
    if i.mode.rounds() && i.lookup_ok && i.confident && i.lookup_len >= LOOKUP_MIN {
        return Choice::Lookup;
    }
    if i.mode.chains() && i.mtp_drafts >= 1 && match_len >= i.min_match && gate().takes() {
        return Choice::Chain;
    }
    Choice::Mtp
}

/// The lookup round length: the largest bucket (or `max_len` itself) not
/// above `min(max_len, slot capacity, remaining - 1)`; 0 below [`LOOKUP_MIN`].
pub(crate) fn lookup_len(max_len: usize, capacity: usize, remaining: usize) -> usize {
    let l = max_len.min(capacity).min(remaining.saturating_sub(1));
    if l < LOOKUP_MIN {
        return 0;
    }
    LOOKUP_BUCKETS
        .iter()
        .copied()
        .chain(std::iter::once(max_len))
        .filter(|&b| b <= l)
        .max()
        .unwrap_or(0)
}

/// What one sequence presents to the planner.
pub(crate) struct RowFacts<'a> {
    pub head: &'a [u32],
    pub last: u32,
    pub no_drafts: bool,
    pub grammar: bool,
    pub mtp_drafts: &'a [u32],
    pub lookup_ok: bool,
    pub capacity: usize,
    pub remaining: usize,
}

/// The planner's result for one sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Plan {
    NoSpec,
    Mtp,
    Chain(Vec<u32>),
    Lookup(Vec<u32>),
}

/// Plan one sequence's round. See the module docs.
pub(crate) fn plan(state: &mut CopyState, cfg: &Config, f: &RowFacts) -> Plan {
    if f.no_drafts {
        return Plan::NoSpec;
    }
    if cfg.mode == Mode::Off || f.grammar {
        return Plan::Mtp;
    }
    state.sync(f.head, Some(f.last));
    let width = f.mtp_drafts.len();
    let lookup_len = if cfg.mode.rounds() && f.lookup_ok {
        lookup_len(cfg.max_len, f.capacity, f.remaining)
    } else {
        0
    };
    let prop = state.propose(lookup_len.max(width));
    let inputs = Inputs {
        mode: cfg.mode,
        no_drafts: false,
        grammar: false,
        mtp_drafts: width,
        lookup_ok: f.lookup_ok,
        lookup_len,
        match_len: prop.as_ref().map(|p| p.match_len),
        confident: prop.as_ref().is_some_and(|p| p.confident),
        min_match: cfg.min_match,
    };
    let mut verdict = None;
    let choice = select(&inputs, || {
        let v = state.gate(width, cfg.advantage);
        verdict = Some(v);
        v
    });
    match verdict {
        Some(GateVerdict::Skip) => add(&METRICS.skipped, 1),
        Some(GateVerdict::Probe) => add(&METRICS.probes, 1),
        _ => {}
    }
    match (choice, prop) {
        (Choice::Chain, Some(p)) => {
            let t = p.tokens[..width].to_vec();
            state.dispatch(Kind::Chain, &t);
            Plan::Chain(t)
        }
        (Choice::Lookup, Some(p)) => {
            let t = p.tokens[..lookup_len].to_vec();
            state.dispatch(Kind::Lookup, &t);
            Plan::Lookup(t)
        }
        (Choice::NoSpec, _) => Plan::NoSpec,
        _ => {
            state.dispatch(Kind::Mtp, f.mtp_drafts);
            Plan::Mtp
        }
    }
}
