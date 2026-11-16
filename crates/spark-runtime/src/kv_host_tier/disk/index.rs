// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the disk tier's in-memory index (lane L27): one entry per record file, the
//! least-recently-used order (by last use, seeded from file mtimes at the restart scan), the byte
//! accounting against the budget, and the keys whose write is in flight.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

/// LRU position: unix nanoseconds of the last use, then a tie-breaker.
pub(super) type Stamp = (u64, u64);

pub(super) fn unix_nanos(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// One record file.
pub(super) struct Entry {
    pub(super) namespace: u64,
    pub(super) payload_len: u64,
    pub(super) file_bytes: u64,
    pub(super) stamp: Stamp,
    /// Changes on every (re)insert, so a reader never removes a record a writer just replaced.
    /// Named `generation` because `gen` is a reserved keyword in edition 2024.
    pub(super) generation: u64,
}

#[derive(Default)]
pub(super) struct State {
    pub(super) entries: HashMap<[u8; 16], Entry>,
    lru: BTreeMap<Stamp, [u8; 16]>,
    /// Bytes of every indexed record plus the reservations of in-flight writes.
    pub(super) used: u64,
    /// The served model's fingerprint, named by `adopt_on_restart`.
    pub(super) namespace: Option<u64>,
    tie: u64,
    generation: u64,
    pub(super) inflight: HashSet<[u8; 16]>,
}

impl State {
    fn stamp(&mut self, nanos: u64) -> Stamp {
        self.tie += 1;
        (nanos, self.tie)
    }

    /// Index a record (replacing any entry of the same key).
    pub(super) fn insert(
        &mut self,
        key: [u8; 16],
        namespace: u64,
        payload_len: u64,
        file_bytes: u64,
        nanos: u64,
    ) {
        let stamp = self.stamp(nanos);
        self.generation += 1;
        let e = Entry {
            namespace,
            payload_len,
            file_bytes,
            stamp,
            generation: self.generation,
        };
        if let Some(old) = self.entries.insert(key, e) {
            self.lru.remove(&old.stamp);
            self.used -= old.file_bytes;
        }
        self.lru.insert(stamp, key);
        self.used += file_bytes;
    }

    /// Make `key` the most recently used record.
    pub(super) fn touch(&mut self, key: &[u8; 16]) {
        let stamp = self.stamp(unix_nanos(SystemTime::now()));
        if let Some(e) = self.entries.get_mut(key) {
            self.lru.remove(&e.stamp);
            e.stamp = stamp;
            self.lru.insert(stamp, *key);
        }
    }

    pub(super) fn remove(&mut self, key: &[u8; 16]) -> Option<Entry> {
        let e = self.entries.remove(key)?;
        self.lru.remove(&e.stamp);
        self.used -= e.file_bytes;
        Some(e)
    }

    /// Remove least recently used entries until `need` more bytes fit `budget`; returns their
    /// keys (the caller deletes the files).
    pub(super) fn victims(&mut self, need: u64, budget: u64) -> Vec<[u8; 16]> {
        let mut out = Vec::new();
        while self.used + need > budget {
            let Some((_, &k)) = self.lru.first_key_value() else {
                break;
            };
            self.remove(&k);
            out.push(k);
        }
        out
    }
}
