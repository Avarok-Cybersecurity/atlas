// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0. Source: src/models/qwen3_5/program/speculative/prompt_lookup.cpp:25-34 (window hash), :36-91 (window tables, incremental indexing), :93-125 (longest-window lookup, match extension, period continuation). Changes: translated from C++ to Rust; a 16-token window added in front of 8/4/2; exact-content collision chains so a hash collision can never hide an occurrence; a bounded index that drops the oldest window ends; truncation of the committed stream rebuilds instead of silently diverging, by Atlas contributors.
// Portions copied and adapted from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce, zig/src/core/copy_index.zig:1-80. Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written before TensorFold 0.6.0 is also available under the MIT License. See COPY-DRAFT-ATTRIBUTION.md. Changes for Atlas: the "latest earlier occurrence of the context's last tokens" contract and its plain-scan oracle, translated from Zig to Rust over hashed windows instead of per-token position lists.

//! Responsibility: the per-sequence copy index over COMMITTED tokens.
//!
//! The index holds the sequence's committed stream (prompt plus every emitted
//! token, never a draft) and, for each window length in [`WINDOWS`], a table
//! from the window's content to the LATEST window END position that holds it.
//! [`CopyIndex::longest`] answers "where did the context's last tokens occur
//! before": the longest window whose suffix occurs earlier wins, its latest
//! occurrence is taken, and the match is then extended backward up to
//! [`MATCH_LIMIT`] tokens. [`CopyIndex::continuation`] reads the tokens that
//! followed the source, repeating with the source's period past the anchor
//! (short periodic content such as `a b a b` copies itself forward).
//!
//! Exactness: a table key is a 64-bit window hash, but every hit is confirmed
//! by an exact compare, and keys that collide keep one entry per distinct
//! content (`overflow`), so a collision can neither produce a wrong source nor
//! hide a real one. The test oracle forces collisions with a tiny key mask.
//!
//! Memory: window ends older than `cap` positions are dropped (`floor`), and
//! the tables are compacted when the floor has moved by a quarter of the cap.
//! The committed tokens themselves are kept (4 bytes each), since the backward
//! extension and the continuation read them.

use std::collections::HashMap;

/// Indexed suffix window lengths, longest first (16 from the KVMem design
/// description, 8/4/2 from Cinference's `kWindows`).
pub(crate) const WINDOWS: [usize; 4] = [16, 8, 4, 2];

/// Backward match extension cap (Cinference `kMatchLimit`).
pub(crate) const MATCH_LIMIT: usize = 128;

/// Default number of window end positions kept per sequence.
pub(crate) const DEFAULT_CAP: usize = 1 << 18;

/// One source: the window that hit (index into [`WINDOWS`]), the latest
/// earlier end position of that window's content, and the number of tokens
/// before and at `end` that equal the context's last tokens (`n >= window`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Match {
    pub window: usize,
    pub end: usize,
    pub n: usize,
}

impl Match {
    /// Position of the first token that followed the occurrence.
    pub fn at(&self) -> usize {
        self.end + 1
    }
}

/// How a [`CopyIndex::sync`] reconciled the committed stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SyncOutcome {
    /// The stream extended the indexed tokens; `from` is the old length, so
    /// the new committed tokens are `tokens()[from..]`.
    Appended { from: usize },
    /// The stream no longer extended the indexed tokens (a rollback rewrote
    /// the tail, or a different sequence): everything was indexed again.
    Rebuilt,
}

#[derive(Default)]
struct WindowTable {
    /// key -> latest end of the first content seen under this key.
    primary: HashMap<u64, u32>,
    /// key -> latest end of each further distinct content under this key
    /// (non-empty only under hash collisions). Invariant: a key present here
    /// is present in `primary`.
    overflow: HashMap<u64, Vec<u32>>,
}

/// Cinference's `window_hash` (`prompt_lookup.cpp:25-34`).
pub(crate) fn window_hash(tokens: &[u32]) -> u64 {
    let mut hash: u64 = 0x9e37_79b9_7f4a_7c15 ^ tokens.len() as u64;
    for &t in tokens {
        hash ^= t as u64;
        hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
        hash ^= hash >> 32;
    }
    hash
}

/// The per-sequence index. See the module docs.
pub(crate) struct CopyIndex {
    toks: Vec<u32>,
    /// Window end positions `[floor, indexed)` are in the tables.
    indexed: usize,
    floor: usize,
    compacted_floor: usize,
    cap: usize,
    mask: u64,
    tables: [WindowTable; WINDOWS.len()],
}

impl Default for CopyIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl CopyIndex {
    pub fn new() -> Self {
        Self::with_params(DEFAULT_CAP, u64::MAX)
    }

    /// `cap` window ends kept (at least 64); `mask` is ANDed into every table
    /// key (`u64::MAX` in production; a tiny mask in tests forces collisions).
    pub fn with_params(cap: usize, mask: u64) -> Self {
        Self {
            toks: Vec::new(),
            indexed: 0,
            floor: 0,
            compacted_floor: 0,
            cap: cap.max(64),
            mask,
            tables: Default::default(),
        }
    }

    /// The committed tokens the index holds.
    pub fn tokens(&self) -> &[u32] {
        &self.toks
    }

    pub fn len(&self) -> usize {
        self.toks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.toks.is_empty()
    }

    /// Oldest window end still indexed.
    pub fn floor(&self) -> usize {
        self.floor
    }

    /// Distinct window entries held (memory accounting and tests).
    pub fn entries(&self) -> usize {
        self.tables
            .iter()
            .map(|t| t.primary.len() + t.overflow.values().map(Vec::len).sum::<usize>())
            .sum()
    }

    fn reset(&mut self) {
        self.toks.clear();
        self.indexed = 0;
        self.floor = 0;
        self.compacted_floor = 0;
        for t in &mut self.tables {
            t.primary.clear();
            t.overflow.clear();
        }
    }

    /// Reconcile with the committed stream `head ++ last` (the scheduler
    /// passes `seq.tokens` and `last_token`: the sampled token is committed
    /// before the forward that feeds it). Appends what is new; a stream that
    /// does not extend the held tokens is indexed again from scratch.
    pub fn sync(&mut self, head: &[u32], last: Option<u32>) -> SyncOutcome {
        let len = head.len() + usize::from(last.is_some());
        let n = self.toks.len();
        let consistent = n <= len
            && if n <= head.len() {
                self.toks[..] == head[..n]
            } else {
                self.toks[..head.len()] == head[..] && Some(self.toks[n - 1]) == last
            };
        let outcome = if consistent {
            SyncOutcome::Appended { from: n }
        } else {
            self.reset();
            SyncOutcome::Rebuilt
        };
        let n = self.toks.len();
        if n < head.len() {
            self.toks.extend_from_slice(&head[n..]);
        }
        if let Some(t) = last
            && self.toks.len() == head.len()
        {
            self.toks.push(t);
        }
        // Windows end strictly before the anchor (the last token), so every
        // hit is an EARLIER occurrence of the suffix.
        let end = self.toks.len().saturating_sub(1);
        self.index_until(end);
        outcome
    }

    fn index_until(&mut self, end: usize) {
        if end <= self.indexed {
            return;
        }
        // A fresh (or rebuilt) long history starts at the cap's horizon
        // instead of inserting ends it would drop at once.
        let horizon = end.saturating_sub(self.cap);
        if self.indexed < horizon {
            self.indexed = horizon;
            self.floor = self.floor.max(horizon);
        }
        for e in self.indexed..end {
            for wi in 0..WINDOWS.len() {
                if e + 1 >= WINDOWS[wi] {
                    self.insert(wi, e);
                }
            }
        }
        self.indexed = end;
        self.floor = self.floor.max(end.saturating_sub(self.cap));
        if self.floor >= self.compacted_floor + self.cap / 4 {
            self.compact();
        }
    }

    fn insert(&mut self, wi: usize, e: usize) {
        let w = WINDOWS[wi];
        let start = e + 1 - w;
        let Self {
            toks, tables, mask, ..
        } = self;
        let key = window_hash(&toks[start..=e]) & *mask;
        let same = |p: u32| {
            let p = p as usize;
            toks[p + 1 - w..=p] == toks[start..=e]
        };
        let t = &mut tables[wi];
        match t.primary.get_mut(&key) {
            None => {
                t.primary.insert(key, e as u32);
            }
            Some(p) => {
                if same(*p) {
                    *p = e as u32;
                } else {
                    let chain = t.overflow.entry(key).or_default();
                    match chain.iter_mut().find(|q| same(**q)) {
                        Some(q) => *q = e as u32,
                        None => chain.push(e as u32),
                    }
                }
            }
        }
    }

    /// Drop every entry whose latest end is below the floor.
    fn compact(&mut self) {
        let floor = self.floor as u32;
        for t in &mut self.tables {
            t.overflow.retain(|_, chain| {
                chain.retain(|&p| p >= floor);
                !chain.is_empty()
            });
            let WindowTable { primary, overflow } = t;
            primary.retain(|key, p| {
                if *p >= floor {
                    return true;
                }
                // Promote a surviving collision entry so the invariant
                // "overflow key => primary key" holds.
                match overflow.get_mut(key) {
                    Some(chain) => {
                        *p = chain.swap_remove(0);
                        if chain.is_empty() {
                            overflow.remove(key);
                        }
                        true
                    }
                    None => false,
                }
            });
        }
        self.compacted_floor = self.floor;
    }

    /// Latest end `>= floor` of a window equal to `toks[s..s + w]`.
    fn find(&self, wi: usize, s: usize) -> Option<usize> {
        let w = WINDOWS[wi];
        let suffix = &self.toks[s..s + w];
        let key = window_hash(suffix) & self.mask;
        let ok = |p: u32| {
            let p = p as usize;
            p >= self.floor && self.toks[p + 1 - w..=p] == *suffix
        };
        let t = &self.tables[wi];
        let p = *t.primary.get(&key)?;
        if ok(p) {
            return Some(p as usize);
        }
        t.overflow
            .get(&key)?
            .iter()
            .copied()
            .find(|&q| ok(q))
            .map(|q| q as usize)
    }

    /// The source for the context's last tokens: the longest window (16, 8,
    /// 4, 2) whose suffix occurs earlier, at its LATEST earlier occurrence,
    /// with the match extended backward up to [`MATCH_LIMIT`] tokens. `None`
    /// when no window's suffix occurs earlier (or the history is too short).
    pub fn longest(&self) -> Option<Match> {
        let len = self.toks.len();
        if len < 2 {
            return None;
        }
        for (wi, &w) in WINDOWS.iter().enumerate() {
            if len < w + 1 {
                continue;
            }
            if let Some(end) = self.find(wi, len - w) {
                return Some(Match {
                    window: wi,
                    end,
                    n: extend_match(&self.toks, end, w),
                });
            }
        }
        None
    }

    /// Up to `want` tokens following position `next - 1` of the committed
    /// stream: the stream's own tokens while they last, then repeating with
    /// period `len - next` (Cinference `prompt_lookup.cpp:118-123`). `next`
    /// must be below `len()`.
    pub fn continuation(&self, next: usize, want: usize, out: &mut Vec<u32>) {
        out.clear();
        let len = self.toks.len();
        if next >= len {
            return;
        }
        let period = len - next;
        for j in 0..want {
            let pos = next + j;
            let t = if pos < len {
                self.toks[pos]
            } else {
                out[j - period]
            };
            out.push(t);
        }
    }

    /// [`Self::continuation`] after a [`Match`].
    pub fn chain(&self, m: &Match, want: usize, out: &mut Vec<u32>) {
        self.continuation(m.at(), want, out);
    }
}

/// Extend a `w`-token match at `end` backward against the context's tail
/// (Cinference `prompt_lookup.cpp:111-115`).
pub(crate) fn extend_match(toks: &[u32], end: usize, w: usize) -> usize {
    let len = toks.len();
    let mut n = w;
    while n < MATCH_LIMIT && n <= end && toks[end - n] == toks[len - 1 - n] {
        n += 1;
    }
    n
}
