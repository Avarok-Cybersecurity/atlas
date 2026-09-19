// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! [`ExpertLru`]'s victim selection: strict LRU, or a random slot among the
//! oldest few percent (the cure for the cyclic replay of a working set a
//! little larger than the cache). Split from `expert_lru.rs` (500-LoC cap).

use anyhow::{Result, bail};

use super::expert_lru::{ExpertLru, NONE};

impl ExpertLru {
    /// Strict LRU or a random victim among the oldest `pct`% (see `take_victim`).
    pub fn set_evict_random_pct(&mut self, pct: usize) {
        self.evict_pct = pct.min(100);
    }

    /// xorshift64*, seeded once: the same run picks the same victims.
    fn rng_next(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Take a victim among the `evict_pct`% least recently used slots that
    /// are not pinned in this epoch and not being filled, dropping whatever
    /// it held. Strict LRU (`evict_pct` = 0, or fewer than 100/pct slots)
    /// takes the oldest; otherwise a random one of the oldest few percent.
    ///
    /// The randomness is the cure for the cyclic case: a request whose
    /// working set is a few experts larger than the cache re-reads it in the
    /// same order, and strict LRU evicts exactly the expert needed next at
    /// every miss (MinHeap at 8,380 slots: 8,418 distinct experts, 5.5
    /// misses a step on r2/r3, 09-19). A victim drawn from the oldest 5%
    /// breaks the lockstep: 1.1 misses a step in the replay, Volvo (fits)
    /// unchanged at 0. The policy never touches the math.
    pub(super) fn take_victim(&mut self) -> Result<u32> {
        let want = (self.n_slots * self.evict_pct / 100).max(1);
        let mut cands: [u32; 8] = [NONE; 8];
        let mut n = 0usize;
        // reservoir-sample up to eight of the oldest `want` eligible slots
        let mut seen = 0usize;
        let mut i = self.tail;
        while i != NONE && seen < want {
            let m = &self.meta[i as usize];
            if m.epoch != self.epoch && m.ticket.is_none() {
                if n < cands.len() {
                    cands[n] = i;
                    n += 1;
                } else {
                    let j = (self.rng_next() % (seen as u64 + 1)) as usize;
                    if j < cands.len() {
                        cands[j] = i;
                    }
                }
                seen += 1;
            }
            i = self.meta[i as usize].prev;
        }
        if n > 0 {
            let pick = if n == 1 {
                cands[0]
            } else {
                cands[(self.rng_next() % n as u64) as usize]
            };
            if let Some(k) = self.meta[pick as usize].key.take() {
                self.map.remove(&k);
                self.stats.evictions += 1;
            }
            return Ok(pick);
        }
        bail!(
            "expert cache of {} slots is smaller than one token's working set ({} pinned)",
            self.n_slots,
            self.map.len()
        )
    }
}
