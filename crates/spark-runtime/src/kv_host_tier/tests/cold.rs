// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the cold-store seam: an evicted block is demoted with its tokens, promoted
//! back on lookup (without evicting the hits that lookup already counted) and restores its
//! bytes; a store that refuses (other tokens, other size, an I/O error) only ever causes a miss;
//! keys already in the store are adopted on attach; a key-only store works through the
//! defaulted token-aware methods.

use std::collections::HashMap;

use anyhow::{Result, bail};
use parking_lot::Mutex;

use super::*;
use crate::kv_host_tier::{ColdStore, RestoreState, TierPolicy};

/// One stored record: the block's tokens and its bytes.
type ColdRecord = (Vec<u32>, Vec<u8>);

/// An in-memory cold store that keeps tokens and checks them on promote.
#[derive(Default)]
struct MemCold {
    recs: Mutex<HashMap<BlockKey, ColdRecord>>,
    fail_reads: std::sync::atomic::AtomicBool,
}

impl MemCold {
    fn get(&self, k: &BlockKey, tokens: Option<&[u32]>, out: &mut [u8]) -> Result<bool> {
        if self.fail_reads.load(std::sync::atomic::Ordering::Relaxed) {
            bail!("simulated read error");
        }
        match self.recs.lock().get(k) {
            Some((t, b)) if b.len() == out.len() && tokens.is_none_or(|want| want == &t[..]) => {
                out.copy_from_slice(b);
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

impl ColdStore for MemCold {
    fn demote(&self, k: &BlockKey, b: &[u8]) -> Result<()> {
        self.recs.lock().insert(*k, (Vec::new(), b.to_vec()));
        Ok(())
    }

    fn promote(&self, k: &BlockKey, out: &mut [u8]) -> Result<bool> {
        self.get(k, None, out)
    }

    fn adopt_on_restart(&self, _ns_fp: u64) -> Result<Vec<BlockKey>> {
        Ok(self.recs.lock().keys().copied().collect())
    }

    fn demote_with_tokens(&self, k: &BlockKey, tokens: &[u32], b: &[u8]) -> Result<()> {
        self.recs.lock().insert(*k, (tokens.to_vec(), b.to_vec()));
        Ok(())
    }

    fn promote_with_tokens(&self, k: &BlockKey, tokens: &[u32], out: &mut [u8]) -> Result<bool> {
        self.get(k, Some(tokens), out)
    }
}

/// A store that implements only the three contract methods (keys and bytes).
#[derive(Default)]
struct KeyOnlyCold(Mutex<HashMap<BlockKey, Vec<u8>>>);

impl ColdStore for KeyOnlyCold {
    fn demote(&self, k: &BlockKey, b: &[u8]) -> Result<()> {
        self.0.lock().insert(*k, b.to_vec());
        Ok(())
    }

    fn promote(&self, k: &BlockKey, out: &mut [u8]) -> Result<bool> {
        match self.0.lock().get(k) {
            Some(b) if b.len() == out.len() => {
                out.copy_from_slice(b);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn adopt_on_restart(&self, _ns_fp: u64) -> Result<Vec<BlockKey>> {
        Ok(self.0.lock().keys().copied().collect())
    }
}

fn attach(r: &Rig, cold: std::sync::Arc<dyn ColdStore>) {
    r.tier.set_namespace(NS);
    r.tier.set_cold_store(cold);
}

/// Budget 10 blocks (past 8, evict down to 6): A (4 blocks), B (4), C (1) evicts a1..a3.
fn evict_a_tail(r: &Rig) -> (Prompt, Vec<Vec<u8>>, Vec<BlockRegions>) {
    let a = Prompt::new(NS, 0, 4);
    let blocks: Vec<BlockRegions> = (0..4).map(|i| device_block(&r.gpu, i + 1)).collect();
    let want: Vec<Vec<u8>> = blocks.iter().map(|b| read_block(&r.gpu, b)).collect();
    store_published(r, &a, &blocks);
    let b = Prompt::new(NS, 1000, 4);
    let bb: Vec<BlockRegions> = (0..4).map(|i| device_block(&r.gpu, 20 + i)).collect();
    store_published(r, &b, &bb);
    store_published(r, &Prompt::new(NS, 2000, 1), &[device_block(&r.gpu, 50)]);
    (a, want, blocks)
}

#[test]
fn an_evicted_block_is_demoted_and_promoted_back() {
    let r = rig(10, TierPolicy::Lru);
    let cold = std::sync::Arc::new(MemCold::default());
    attach(&r, cold.clone());
    let (a, want, blocks) = evict_a_tail(&r);
    let s = r.tier.snapshot();
    assert_eq!(s.evictions, 3, "A's tail a1..a3 left");
    assert_eq!(s.cold_demoted, 3);
    assert_eq!(cold.recs.lock().len(), 3);
    // A lookup promotes the demoted tail back and counts every block; the third promotion
    // needs room, and the hits already counted (a0..a2) are never what makes it.
    assert_eq!(r.tier.lookup(&a.keys, &a.chunks()), 4);
    assert_eq!(r.tier.snapshot().cold_promoted, 3);
    assert_eq!(
        r.tier.lookup(&a.keys, &a.chunks()),
        4,
        "all four still in host RAM"
    );
    for blk in &blocks {
        clobber(&r.gpu, blk);
    }
    let dst: Vec<BlockRegions> = (0..4).map(|_| empty_block(&r.gpu)).collect();
    let t = r.tier.begin_restore(r.gpu.as_ref(), &a.keys, &dst).unwrap();
    assert_eq!(
        r.tier.poll(r.gpu.as_ref(), &t).unwrap(),
        RestoreState::Ready
    );
    r.tier.release(t);
    for (d, w) in dst.iter().zip(&want) {
        assert_eq!(&read_block(&r.gpu, d), w);
    }
}

#[test]
fn a_refusing_store_only_ever_causes_a_miss() {
    let r = rig(64, TierPolicy::Lru);
    let cold = std::sync::Arc::new(MemCold::default());
    let toks: Vec<u32> = (0..BS as u32).collect();
    let k = BlockKey::chain(NS, 0, None, &toks);
    // Other tokens stored under the key.
    let other: Vec<u32> = toks.iter().map(|t| t + 1).collect();
    cold.demote_with_tokens(&k, &other, &[7u8; BLOCK_BYTES])
        .unwrap();
    attach(&r, cold.clone());
    assert_eq!(r.tier.lookup(&[k], &[&toks]), 0, "tokens differ");
    // A block of another size.
    cold.demote_with_tokens(&k, &toks, &[7u8; BLOCK_BYTES - 4])
        .unwrap();
    attach(&r, cold.clone());
    assert_eq!(r.tier.lookup(&[k], &[&toks]), 0, "wrong length");
    // A read error.
    cold.demote_with_tokens(&k, &toks, &[7u8; BLOCK_BYTES])
        .unwrap();
    cold.fail_reads
        .store(true, std::sync::atomic::Ordering::Relaxed);
    attach(&r, cold.clone());
    assert_eq!(r.tier.lookup(&[k], &[&toks]), 0, "read error");
    assert_eq!(r.tier.snapshot().cold_promoted, 0);
    assert_eq!(r.arena.live(), 0, "no refused block holds a host run");
    // Readable again: the same record now serves.
    cold.fail_reads
        .store(false, std::sync::atomic::Ordering::Relaxed);
    attach(&r, cold);
    assert_eq!(r.tier.lookup(&[k], &[&toks]), 1);
}

#[test]
fn keys_in_the_store_are_adopted_on_attach() {
    let r = rig(64, TierPolicy::Lru);
    let cold = std::sync::Arc::new(MemCold::default());
    let p = Prompt::new(NS, 10, 2);
    for (i, (k, t)) in p.keys.iter().zip(p.chunks()).enumerate() {
        cold.demote_with_tokens(k, t, &[i as u8 + 1; BLOCK_BYTES])
            .unwrap();
    }
    attach(&r, cold);
    assert_eq!(
        r.tier.lookup(&p.keys, &p.chunks()),
        2,
        "a warm restart serves the blocks"
    );
    // Another model's keys differ (the namespace is chained into every key).
    let stale = Prompt::new(NS + 9, 10, 2);
    assert_eq!(r.tier.lookup(&stale.keys, &stale.chunks()), 0);
    let dst = [empty_block(&r.gpu)];
    let t = r
        .tier
        .begin_restore(r.gpu.as_ref(), &p.keys[1..], &dst)
        .unwrap();
    r.tier.wait(r.gpu.as_ref(), &t).unwrap();
    r.tier.release(t);
    assert_eq!(read_block(&r.gpu, &dst[0]), vec![2u8; BLOCK_BYTES]);
}

#[test]
fn a_key_only_store_works_through_the_defaults() {
    let r = rig(10, TierPolicy::Lru);
    let cold = std::sync::Arc::new(KeyOnlyCold::default());
    attach(&r, cold.clone());
    let (a, want, _blocks) = evict_a_tail(&r);
    assert_eq!(cold.0.lock().len(), 3);
    assert_eq!(r.tier.lookup(&a.keys, &a.chunks()), 4);
    let dst: Vec<BlockRegions> = (0..3).map(|_| empty_block(&r.gpu)).collect();
    let t = r
        .tier
        .begin_restore(r.gpu.as_ref(), &a.keys[1..], &dst)
        .unwrap();
    r.tier.wait(r.gpu.as_ref(), &t).unwrap();
    r.tier.release(t);
    for (d, w) in dst.iter().zip(&want[1..]) {
        assert_eq!(&read_block(&r.gpu, d), w);
    }
}
