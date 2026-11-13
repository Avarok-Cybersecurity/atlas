// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: count the committed tokens the MTP head sees and write a
//! rank file from them (`AVAROK_MTP_VOCAB_RANK_RECORD=<file>`), which is how
//! Atlas produces its own draft-vocabulary ranking through the real server to
//! replace the Cinference-derived v1 table.
//!
//! The head sees every committed token of a speculating sequence without any
//! extra readback: a propose at position p carries the last committed token
//! (the verify's bonus token), and the round before it proposed drafts
//! d1..dk at p0; the verify accepted `p - p0 - 1` of them. So at each propose
//! the recorder counts the accepted prefix of the previous round's drafts plus
//! the new token. A gap (the sequence decoded serially, so `p - p0 - 1`
//! exceeds the drafts) counts the new token only. Per-sequence entries are
//! keyed by the proposer state's address and dropped when the state is freed
//! or its KV handed off.
//!
//! Output, written atomically (`.tmp` then rename) every
//! [`FLUSH_EVERY`] counted tokens and when a sequence ends at least
//! [`FLUSH_SECS`] seconds after the last write:
//! * `<file>`: u32 little-endian token ids, most counted first, ties by
//!   ascending id, ids never seen omitted (the layout of
//!   `data/mtp_vocab_rank_qwen35.u32`);
//! * `<file>.counts.u64`: u64 little-endian count per token id (sum several
//!   runs before ranking);
//! * `<file>.json`: totals and the vocabulary size.
//!
//! Off (no allocation, one `OnceLock` read per propose) unless the variable
//! is set. It only observes; drafts and outputs are unchanged.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Environment variable naming the output rank file.
pub const ENV: &str = "AVAROK_MTP_VOCAB_RANK_RECORD";
/// Counted tokens between two writes.
pub const FLUSH_EVERY: u64 = 65_536;
/// Minimum seconds between two end-of-sequence writes.
pub const FLUSH_SECS: u64 = 30;

/// Committed-token counts plus the per-sequence round memory.
#[derive(Debug)]
pub struct Counts {
    counts: Vec<u64>,
    total: u64,
    since_flush: u64,
    /// state key -> (position of its last propose, the drafts it returned)
    pending: HashMap<usize, (usize, Vec<u32>)>,
}

impl Counts {
    pub fn new(vocab: usize) -> Self {
        Self {
            counts: vec![0; vocab],
            total: 0,
            since_flush: 0,
            pending: HashMap::new(),
        }
    }

    fn bump(&mut self, id: u32) {
        if let Some(c) = self.counts.get_mut(id as usize) {
            *c += 1;
            self.total += 1;
            self.since_flush += 1;
        }
    }

    /// A propose of sequence `key` at `position` whose last committed token
    /// is `last_token`: count the previous round's accepted drafts and it.
    pub fn on_propose(&mut self, key: usize, position: usize, last_token: u32) {
        if let Some((p0, drafts)) = self.pending.remove(&key)
            && position > p0
        {
            let accepted = position - p0 - 1;
            if accepted <= drafts.len() {
                for &d in &drafts[..accepted] {
                    self.bump(d);
                }
            }
        }
        self.bump(last_token);
    }

    /// Remember the drafts the propose at `position` returned for `key`.
    pub fn on_drafts(&mut self, key: usize, position: usize, drafts: &[u32]) {
        self.pending.insert(key, (position, drafts.to_vec()));
    }

    /// The sequence behind `key` ended (or handed off its drafter KV).
    pub fn forget(&mut self, key: usize) {
        self.pending.remove(&key);
    }

    /// Ids by descending count, ties by ascending id, unseen ids omitted.
    pub fn ranking(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..self.counts.len() as u32)
            .filter(|&i| self.counts[i as usize] > 0)
            .collect();
        ids.sort_by(|&a, &b| {
            self.counts[b as usize]
                .cmp(&self.counts[a as usize])
                .then(a.cmp(&b))
        });
        ids
    }

    #[cfg(test)]
    pub fn count(&self, id: u32) -> u64 {
        self.counts.get(id as usize).copied().unwrap_or(0)
    }

    #[cfg(test)]
    pub fn total(&self) -> u64 {
        self.total
    }
}

struct Recorder {
    path: PathBuf,
    state: Mutex<(Counts, Instant)>,
}

static RECORDER: OnceLock<Option<Recorder>> = OnceLock::new();

/// Arm the recorder for a vocabulary of `vocab` ids when the variable is set
/// (first call wins; later models keep the first vocabulary).
pub(super) fn init(vocab: usize) {
    RECORDER.get_or_init(|| {
        let path = std::env::var_os(ENV).filter(|p| !p.is_empty())?;
        let path = PathBuf::from(path);
        tracing::info!(
            "MTP vocab recorder: counting committed tokens into {} (vocab {vocab})",
            path.display()
        );
        Some(Recorder {
            path,
            state: Mutex::new((Counts::new(vocab), Instant::now())),
        })
    });
}

fn with<R>(f: impl FnOnce(&Recorder, &mut Counts, &mut Instant) -> R) -> Option<R> {
    let r = RECORDER.get()?.as_ref()?;
    let mut g = r.state.lock().unwrap_or_else(|e| e.into_inner());
    let (counts, last) = &mut *g;
    Some(f(r, counts, last))
}

/// Hook: a propose of the sequence whose state lives at `key`.
pub(super) fn on_propose(key: usize, position: usize, last_token: u32) {
    with(|r, c, last| {
        c.on_propose(key, position, last_token);
        if c.since_flush >= FLUSH_EVERY {
            write_files(r, c, last);
        }
    });
}

/// Hook: the drafts that propose returned (only real ids, never the
/// deferred-readback placeholder).
pub(super) fn on_drafts(key: usize, position: usize, drafts: &[u32]) {
    with(|_, c, _| c.on_drafts(key, position, drafts));
}

/// Hook: the sequence's proposer state was freed or its KV taken.
pub(super) fn forget(key: usize) {
    with(|r, c, last| {
        c.forget(key);
        if c.since_flush > 0 && last.elapsed().as_secs() >= FLUSH_SECS {
            write_files(r, c, last);
        }
    });
}

fn write_files(r: &Recorder, c: &mut Counts, last: &mut Instant) {
    let ranking: Vec<u8> = c.ranking().iter().flat_map(|i| i.to_le_bytes()).collect();
    let counts: Vec<u8> = c.counts.iter().flat_map(|n| n.to_le_bytes()).collect();
    let meta = serde_json::json!({
        "artifact": "mtp_vocab_rank_recording",
        "layout": "u32 little-endian token ids, most counted first, unseen ids omitted",
        "counts_layout": "u64 little-endian count per token id",
        "vocab": c.counts.len(),
        "committed_tokens": c.total,
        "distinct": c.counts.iter().filter(|&&n| n > 0).count(),
        "source": "committed tokens seen at MTP proposes (layers/mtp_head/vocab_recorder.rs)",
    });
    let mut ok = true;
    for (suffix, bytes) in [
        ("", ranking),
        (".counts.u64", counts),
        (
            ".json",
            serde_json::to_vec_pretty(&meta).unwrap_or_default(),
        ),
    ] {
        let mut dst = r.path.clone().into_os_string();
        dst.push(suffix);
        let dst = PathBuf::from(dst);
        let mut tmp = dst.clone().into_os_string();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        if let Err(e) = std::fs::write(&tmp, &bytes).and_then(|_| std::fs::rename(&tmp, &dst)) {
            tracing::warn!("MTP vocab recorder: writing {} failed: {e}", dst.display());
            ok = false;
        }
    }
    if ok {
        c.since_flush = 0;
    }
    *last = Instant::now();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_round_counts_the_accepted_drafts_and_the_bonus_token() {
        let mut c = Counts::new(100);
        c.on_propose(1, 10, 5); // first propose: only the committed token
        c.on_drafts(1, 10, &[7, 8, 9]);
        // Verify accepted 2 (7, 8) and emitted bonus 42: next propose at 13.
        c.on_propose(1, 13, 42);
        assert_eq!(
            (c.count(5), c.count(7), c.count(8), c.count(9), c.count(42)),
            (1, 1, 1, 0, 1)
        );
        assert_eq!(c.total(), 4);
    }

    #[test]
    fn a_full_reject_counts_only_the_correction() {
        let mut c = Counts::new(100);
        c.on_drafts(3, 20, &[1, 2]);
        c.on_propose(3, 21, 9);
        assert_eq!((c.count(1), c.count(2), c.count(9)), (0, 0, 1));
    }

    #[test]
    fn a_serial_gap_counts_only_the_new_token() {
        let mut c = Counts::new(100);
        c.on_drafts(4, 20, &[1, 2]);
        // Serial decode in between: 6 positions later, more than the drafts.
        c.on_propose(4, 26, 9);
        assert_eq!((c.count(1), c.count(2), c.count(9)), (0, 0, 1));
    }

    #[test]
    fn sequences_are_independent_and_forget_drops_the_round() {
        let mut c = Counts::new(100);
        c.on_drafts(1, 10, &[3]);
        c.on_drafts(2, 50, &[4]);
        c.forget(1);
        c.on_propose(1, 12, 6); // a new sequence at a recycled key
        c.on_propose(2, 52, 7);
        assert_eq!(
            (c.count(3), c.count(4), c.count(6), c.count(7)),
            (0, 1, 1, 1)
        );
    }

    #[test]
    fn ids_past_the_vocabulary_are_ignored() {
        let mut c = Counts::new(10);
        c.on_propose(1, 1, 10);
        c.on_propose(1, 2, 99);
        assert_eq!(c.total(), 0);
    }

    #[test]
    fn ranking_is_descending_count_then_ascending_id() {
        let mut c = Counts::new(10);
        for t in [4, 4, 4, 2, 2, 7, 7, 1] {
            c.on_propose(t as usize, 1, t);
        }
        assert_eq!(c.ranking(), vec![4, 2, 7, 1]);
    }
}
