// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! [`ExpertLru`] on the reader pool: misses and predicted experts read by
//! the pool's threads into slots that carry a completion ticket, a real
//! fetch waiting only on the slots it needs. Split from `expert_lru.rs`
//! (500-LoC cap).
//!
//! Protocol: `fetch_many_prefetching(keys, predicted)` assigns slots for the
//! keys' misses and enqueues their reads URGENT, then assigns slots for the
//! predicted experts not resident and enqueues those BACKGROUND, then waits
//! for the keys' tickets (a key already in flight from an earlier prediction
//! is waited on too, and counts as a hit for the statistics with `waited`
//! recorded). Every slot touched is pinned in the current epoch, so the GPU
//! launches queued for this token can never see a slot being rewritten. At
//! the next `begin_token`, finished tickets are cleared and predictions
//! nobody asked for are demoted to the LRU end.

use std::io::Write;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use super::expert_lru::{ExpertLru, ExpertSlot};
use super::expert_prefetch::{ReaderPool, Ticket};
use super::expert_stream::ExpertSource;

impl ExpertLru {
    /// Read misses (and predictions) on `threads` persistent readers from
    /// `src` instead of scoped threads per fetch.
    pub fn set_pool(&mut self, src: Arc<dyn ExpertSource + Send + Sync>, threads: usize) {
        self.pool = Some((src, ReaderPool::new(threads)));
    }

    pub fn has_pool(&self) -> bool {
        self.pool.is_some()
    }

    /// Enqueue the read of `key` into slot `i` as `parts` byte ranges.
    fn enqueue_read(&mut self, i: u32, key: (u32, u32), parts: usize, urgent: bool) -> Arc<Ticket> {
        let (src, pool) = self.pool.as_ref().expect("reader pool");
        let bytes = self.layout().bytes;
        let parts = parts.clamp(1, (bytes / (1 << 20)).max(1));
        let per = bytes.div_ceil(parts);
        let ticket = Ticket::new(parts);
        let p = self.slot_ptr(i);
        for j in 0..parts {
            let off = j * per;
            let len = per.min(bytes - off);
            let (src, ticket, p) = (src.clone(), ticket.clone(), p);
            pool.submit(
                Box::new(move || {
                    let base = p.get();
                    // SAFETY: slot `i` is mapped to `key` alone until its
                    // ticket completes, and the parts of one slot are disjoint.
                    let dst = unsafe { std::slice::from_raw_parts_mut(base.add(off), len) };
                    ticket.finish(src.read_expert_range(key.0, key.1, off, dst));
                }),
                urgent,
            );
        }
        self.meta[i as usize].ticket = Some(ticket.clone());
        self.in_flight.push(i);
        ticket
    }

    /// Slots for `keys` (misses read urgently), reads of `predicted` experts
    /// started in the background, then the wait for `keys` only.
    pub fn fetch_many_prefetching(
        &mut self,
        keys: &[(u32, u32)],
        predicted: &[(u32, u32)],
    ) -> Result<Vec<ExpertSlot>> {
        ensure!(self.pool.is_some(), "fetch_many_prefetching without a pool");
        let threads = self.pool.as_ref().map(|p| p.1.threads()).unwrap_or(1);
        let evictions0 = self.stats.evictions;
        let mut out: Vec<u32> = Vec::with_capacity(keys.len());
        let mut misses: Vec<(u32, (u32, u32))> = Vec::new();
        let mut waits: Vec<(u32, Arc<Ticket>)> = Vec::new();
        for &key in keys {
            if let Some(&i) = self.map.get(&key) {
                self.touch(i);
                self.meta[i as usize].speculative = false;
                if let Some(t) = self.meta[i as usize].ticket.clone()
                    && !t.is_done()
                {
                    waits.push((i, t));
                }
                self.stats.hits += 1;
                out.push(i);
                continue;
            }
            let i = self.assign(key)?;
            misses.push((i, key));
            out.push(i);
        }
        let n_wait = waits.len();
        // the misses first, split so all readers work on them
        let parts = if misses.is_empty() {
            1
        } else {
            (threads / misses.len()).clamp(1, 8)
        };
        for &(i, key) in &misses {
            let t = self.enqueue_read(i, key, parts, true);
            waits.push((i, t));
        }
        // then the predictions, whole experts in four ranges, background
        let mut n_pred = 0usize;
        for &key in predicted {
            if self.map.contains_key(&key) {
                continue;
            }
            let Ok(i) = self.assign(key) else { break };
            self.meta[i as usize].speculative = true;
            self.enqueue_read(i, key, 4, false);
            n_pred += 1;
        }
        self.stats.prefetched += n_pred as u64;
        let bytes = self.layout().bytes;
        let mut failed: Vec<(u32, String)> = Vec::new();
        for (i, t) in &waits {
            if let Err(e) = t.wait() {
                failed.push((*i, format!("{e:#}")));
            }
        }
        let n_ok = misses.len()
            - failed
                .iter()
                .filter(|(i, _)| misses.iter().any(|m| m.0 == *i))
                .count();
        self.stats.misses += n_ok as u64;
        self.stats.bytes_read += (n_ok * bytes) as u64;
        self.stats.waited += n_wait as u64;
        if self.trace.is_some() {
            let ev = self.stats.evictions - evictions0;
            self.trace_line(keys, misses.len(), ev, n_wait);
        }
        if let Some((_, first)) = failed.first() {
            let msg = format!(
                "{} of {} expert reads failed; first: {first}",
                failed.len(),
                waits.len()
            );
            for (i, _) in &failed {
                self.unmap(*i);
            }
            bail!(msg);
        }
        Ok(out.into_iter().map(|i| self.slot(i)).collect())
    }

    /// Start reading `predicted` experts in the background (no wait).
    pub fn prefetch(&mut self, predicted: &[(u32, u32)]) -> Result<()> {
        ensure!(self.pool.is_some(), "prefetch without a pool");
        let mut n = 0usize;
        for &key in predicted {
            if self.map.contains_key(&key) {
                continue;
            }
            let Ok(i) = self.assign(key) else { break };
            self.meta[i as usize].speculative = true;
            self.enqueue_read(i, key, 4, false);
            n += 1;
        }
        self.stats.prefetched += n as u64;
        Ok(())
    }

    /// Block until every read in flight has landed (tests, shutdown).
    pub fn wait_in_flight(&self) {
        for &i in &self.in_flight {
            if let Some(t) = &self.meta[i as usize].ticket {
                let _ = t.wait();
            }
        }
    }

    /// At a new token: drop the tickets of finished reads (unmapping the
    /// slots of failed ones) and demote the predictions nobody asked for.
    pub(super) fn reap(&mut self) {
        let mut still = Vec::new();
        for i in std::mem::take(&mut self.in_flight) {
            let Some(t) = self.meta[i as usize].ticket.clone() else {
                continue;
            };
            if !t.is_done() {
                still.push(i);
                continue;
            }
            self.meta[i as usize].ticket = None;
            if t.error().is_some() {
                self.unmap(i);
                continue;
            }
            if self.meta[i as usize].speculative {
                self.meta[i as usize].speculative = false;
                self.stats.prefetch_unused += 1;
                self.demote(i);
            }
        }
        self.in_flight = still;
    }

    // ── diagnostics: the route trace ──

    /// Append every `fetch_many` to `path` as one line: microseconds since
    /// the cache was built, the layer, the key count, the misses and the
    /// evictions of the call, then the expert ids in request order. The
    /// exact access sequence, for replaying cache policies offline.
    pub fn set_trace(&mut self, path: &str) -> Result<()> {
        let f = std::fs::File::create(path).with_context(|| format!("route trace {path}"))?;
        self.trace = Some(std::io::BufWriter::with_capacity(1 << 16, f));
        Ok(())
    }

    pub(super) fn trace_line(
        &mut self,
        keys: &[(u32, u32)],
        misses: usize,
        evictions: u64,
        wait: usize,
    ) {
        let Some(w) = self.trace.as_mut() else {
            return;
        };
        let us = self.t0.elapsed().as_micros();
        let layer = keys.first().map(|k| k.0).unwrap_or(u32::MAX);
        let (pf, pfu) = (self.stats.prefetched, self.stats.prefetch_unused);
        let _ = write!(
            w,
            "{us} L{layer} n={} miss={misses} evict={evictions} wait={wait} pf={pf} pfu={pfu}",
            keys.len()
        );
        for k in keys {
            let _ = write!(w, " {}", k.1);
        }
        let _ = writeln!(w);
        // a killed serve keeps every line written so far
        let _ = w.flush();
    }
}
