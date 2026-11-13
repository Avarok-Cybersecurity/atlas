// SPDX-License-Identifier: AGPL-3.0-only
// Design translated from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08), vllm/v1/kv_offload/cpu/policies/arc.py:28-60 (T1/T2 resident lists, B1/B2 ghost lists, adaptive target), Copyright contributors to the vLLM project, Apache-2.0; no code copied. Algorithm: N. Megiddo and D. S. Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache", FAST 2003. Changes: admission and replacement decoupled (the tier evicts in watermark batches, not on every miss), replacement skips keys a DMA may touch, a batch is applied tail-first.

//! Responsibility: the ARC eviction policy of the host tier.
//!
//! * `T1` holds keys seen once since they entered, `T2` keys hit again (a restore used them).
//! * `B1` and `B2` remember keys recently evicted from `T1` and `T2` (keys only, no bytes).
//! * The target size `p` of `T1` adapts: a re-admission of a key remembered in `B1` means `T1`
//!   was too small (`p` grows by `max(1, |B2| / |B1|)`), one remembered in `B2` means `T2` was
//!   too small (`p` shrinks by `max(1, |B1| / |B2|)`). A key re-admitted from a ghost list
//!   enters `T2`.
//! * Replacement takes the least recent evictable key of `T1` when `|T1| > p` (or `T2` is empty),
//!   else of `T2`, falling back to the other list when the preferred one has no evictable key.
//! * Ghost bounds: `|T1| + |B1| <= c` and `|T1| + |T2| + |B1| + |B2| <= 2c`, with `c` the number
//!   of blocks the byte budget holds.
//!
//! Unlike textbook ARC, admitting a key never evicts here: eviction is driven by the byte
//! watermark through [`ArcPolicy::victims`], and the byte budget keeps `|T1| + |T2| <= c`.

use super::super::fingerprint::BlockKey;
use super::Recency;

#[derive(Debug)]
pub(crate) struct ArcPolicy {
    c: usize,
    p: usize,
    t1: Recency,
    t2: Recency,
    b1: Recency,
    b2: Recency,
}

impl ArcPolicy {
    pub(crate) fn new(capacity_blocks: usize) -> Self {
        Self {
            c: capacity_blocks.max(1),
            p: 0,
            t1: Recency::default(),
            t2: Recency::default(),
            b1: Recency::default(),
            b2: Recency::default(),
        }
    }

    #[cfg(test)]
    pub(crate) fn resident_len(&self) -> usize {
        self.t1.len() + self.t2.len()
    }

    /// The adaptive target size of `T1`, for tests.
    #[cfg(test)]
    pub(crate) fn target(&self) -> usize {
        self.p
    }

    /// `(|T1|, |T2|, |B1|, |B2|)`, for tests.
    #[cfg(test)]
    pub(crate) fn sizes(&self) -> (usize, usize, usize, usize) {
        (self.t1.len(), self.t2.len(), self.b1.len(), self.b2.len())
    }

    /// Lists holding `k`: `"t1"`, `"t2"`, `"b1"`, `"b2"` or `"none"`, for tests.
    #[cfg(test)]
    pub(crate) fn where_is(&self, k: &BlockKey) -> &'static str {
        if self.t1.contains(k) {
            "t1"
        } else if self.t2.contains(k) {
            "t2"
        } else if self.b1.contains(k) {
            "b1"
        } else if self.b2.contains(k) {
            "b2"
        } else {
            "none"
        }
    }

    /// A key became resident (its block was published or promoted).
    pub(crate) fn admit(&mut self, k: BlockKey) {
        if self.t1.contains(&k) || self.t2.contains(&k) {
            self.hit(k);
            return;
        }
        if self.b1.contains(&k) {
            let delta = (self.b2.len() / self.b1.len()).max(1);
            self.p = (self.p + delta).min(self.c);
            self.b1.remove(&k);
            self.t2.push_mru(k);
        } else if self.b2.contains(&k) {
            let delta = (self.b1.len() / self.b2.len()).max(1);
            self.p = self.p.saturating_sub(delta);
            self.b2.remove(&k);
            self.t2.push_mru(k);
        } else {
            self.t1.push_mru(k);
        }
        self.trim_ghosts();
    }

    /// A resident key was used again.
    pub(crate) fn hit(&mut self, k: BlockKey) {
        if self.t1.remove(&k) || self.t2.contains(&k) {
            self.t2.push_mru(k);
        } else {
            self.admit(k);
        }
    }

    /// `k` left without an eviction decision: drop it from the resident lists, keep no ghost.
    pub(crate) fn forget(&mut self, k: &BlockKey) {
        if !self.t1.remove(k) {
            self.t2.remove(k);
        }
    }

    pub(crate) fn victims(&mut self, want: usize, ok: &dyn Fn(&BlockKey) -> bool) -> Vec<BlockKey> {
        let mut out = Vec::with_capacity(want);
        while out.len() < want {
            let t1_first = !self.t1.is_empty() && (self.t1.len() > self.p || self.t2.is_empty());
            let pick = if t1_first {
                self.take(true, ok).or_else(|| self.take(false, ok))
            } else {
                self.take(false, ok).or_else(|| self.take(true, ok))
            };
            match pick {
                Some(k) => out.push(k),
                None => break,
            }
        }
        self.trim_ghosts();
        out
    }

    /// Evict the least recent evictable key of `T1` (`from_t1`) or `T2` into its ghost list.
    fn take(&mut self, from_t1: bool, ok: &dyn Fn(&BlockKey) -> bool) -> Option<BlockKey> {
        let (list, ghost) = if from_t1 {
            (&mut self.t1, &mut self.b1)
        } else {
            (&mut self.t2, &mut self.b2)
        };
        let k = list.lru_where(ok)?;
        list.remove(&k);
        ghost.push_mru(k);
        Some(k)
    }

    fn trim_ghosts(&mut self) {
        while self.t1.len() + self.b1.len() > self.c && self.b1.pop_lru().is_some() {}
        while self.t1.len() + self.t2.len() + self.b1.len() + self.b2.len() > 2 * self.c {
            if self.b2.pop_lru().is_none() && self.b1.pop_lru().is_none() {
                break;
            }
        }
    }
}
