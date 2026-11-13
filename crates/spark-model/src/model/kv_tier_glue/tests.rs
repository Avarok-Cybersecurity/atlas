// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU oracles of the host tier glue on the mock backend (device memory held as
//! host buffers), a real `PagedKvCache`, a real `RadixTree` and lane L26a's tier over an
//! in-memory host arena: write-through, forced radix eviction and a restore that brings back
//! identical bytes; a D2H in flight cannot let an eviction recycle its block; declines (short
//! span, missing snapshot, no free blocks); a namespace mismatch misses. The shared rig is
//! `pub(super)` for the async-path tests (`tests_async.rs`) and the lazy-mode tests
//! (`tests_lazy.rs`).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::kv_host_tier::{HostArena, HostRun, KvHostTier, TierConfig, TierOptions};
use spark_runtime::prefix_cache::PrefixCache;
use spark_runtime::radix_tree::RadixTree;

use super::plan::PrefetchOutcome;
use super::restore::Env;
use super::state::{Live, namespace_from};
use super::store::new_block_positions;
use crate::model::block_mgmt::{apply_evicted_blocks, cache_acquires_refs};

pub(super) const BS: usize = 16;

/// Host runs as heap buffers (stable addresses until freed).
#[derive(Default)]
pub(super) struct VecArena {
    runs: Mutex<HashMap<u64, Box<[u8]>>>,
    next: AtomicU64,
}

impl HostArena for VecArena {
    fn alloc(&self, bytes: usize) -> Option<HostRun> {
        let mut buf = vec![0u8; bytes].into_boxed_slice();
        let ptr = buf.as_mut_ptr() as usize;
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        self.runs.lock().insert(id, buf);
        Some(HostRun { ptr, bytes, id })
    }
    fn free(&self, run: HostRun) {
        self.runs.lock().remove(&run.id);
    }
}

/// Mock backend + KV cache + radix tree + a serving tier.
pub(super) struct Rig {
    pub(super) gpu: MockGpuBackend,
    pub(super) kv: PagedKvCache,
    pub(super) cache: RadixTree,
    pub(super) live: Live,
}

pub(super) fn kv_config() -> KvCacheConfig {
    KvCacheConfig {
        block_size: BS,
        num_kv_heads: 1,
        head_dim: 16,
        num_layers: 2,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

impl Rig {
    /// A rig with `num_blocks` GPU blocks and a tier budget of `budget_blocks` blocks; the
    /// tier's minimum restore is one block.
    pub(super) fn new(num_blocks: usize, budget_blocks: u64, opts: TierOptions) -> Self {
        Self::with(num_blocks, budget_blocks, BS as u32, opts)
    }

    pub(super) fn with(
        num_blocks: usize,
        budget_blocks: u64,
        min_tokens: u32,
        opts: TierOptions,
    ) -> Self {
        let gpu = MockGpuBackend::new();
        let kv = PagedKvCache::new(kv_config(), num_blocks, &gpu).expect("mock kv cache");
        let block_bytes = kv.block_regions(0).bytes();
        let cfg = TierConfig {
            budget_bytes: budget_blocks * block_bytes as u64,
            min_tokens,
            ..TierConfig::default()
        };
        let copy_stream = gpu.create_copy_stream().expect("copy stream");
        let arena: Arc<dyn HostArena> = Arc::new(VecArena::default());
        let opts = TierOptions {
            register_metrics: false,
            ..opts
        };
        let tier = KvHostTier::with_arena(cfg, arena, block_bytes, copy_stream, opts);
        let live = Live::new(Arc::new(tier), 0x5eed, BS, None);
        Self {
            gpu,
            kv,
            cache: RadixTree::new(),
            live,
        }
    }

    pub(super) fn eager() -> Self {
        Self::new(64, 256, TierOptions::default())
    }

    /// Write a recognizable pattern into every region of `block`.
    pub(super) fn fill(&self, block: u32, seed: usize) {
        for (r, (ptr, n)) in self.kv.block_regions(block).regions.into_iter().enumerate() {
            let data: Vec<u8> = (0..n).map(|k| (seed * 131 + r * 7 + k) as u8).collect();
            self.gpu.copy_h2d(&data, ptr).unwrap();
        }
    }

    pub(super) fn read(&self, block: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for (ptr, n) in self.kv.block_regions(block).regions {
            let mut buf = vec![0u8; n];
            self.gpu.copy_d2h(ptr, &mut buf).unwrap();
            out.extend(buf);
        }
        out
    }

    /// A prefill of `tokens`: fresh blocks with distinct bytes, the radix insert and the store
    /// hook. With `finish` the sequence then ends (its node and block references go). Returns
    /// the block table and every block's bytes.
    pub(super) fn prefill(&mut self, tokens: &[u32], finish: bool) -> (Vec<u32>, Vec<Vec<u8>>) {
        let full = tokens.len() / BS;
        let mut table = Vec::new();
        for i in 0..full {
            let b = self.kv.alloc_block().unwrap();
            self.fill(b, tokens[i * BS] as usize + i);
            table.push(b);
        }
        let bytes: Vec<Vec<u8>> = table.iter().map(|&b| self.read(b)).collect();
        let acquired = self
            .cache
            .insert(&tokens[..full * BS], &table, &[], BS, 0, 0);
        cache_acquires_refs(&acquired, &mut self.kv);
        self.live
            .store_inserted(
                &self.gpu,
                &mut self.kv,
                tokens,
                &table,
                0,
                &acquired.blocks,
                0,
            )
            .unwrap();
        if finish {
            self.cache.release(&tokens[..full * BS], BS, 0);
            for &b in &table {
                self.kv.dec_ref(b);
            }
        }
        (table, bytes)
    }

    /// Evict every evictable radix node and return the blocks.
    pub(super) fn evict_all(&mut self) {
        let ev = self.cache.evict(1 << 20);
        apply_evicted_blocks(ev, &mut self.kv);
    }

    /// Overwrite every free block, so restored bytes can only come from the host tier.
    pub(super) fn scribble_free(&mut self) {
        let mut taken = Vec::new();
        while let Some(b) = self.kv.try_alloc_block() {
            self.fill(b, 251);
            taken.push(b);
        }
        for b in taken {
            self.kv.dec_ref(b);
        }
    }

    pub(super) fn reap(&mut self) {
        self.live.reap(&self.gpu, &mut self.kv);
    }

    pub(super) fn prefetch(&mut self, tokens: &[u32], has_ssm: bool) -> super::TierPrefetch {
        let mut env = Env {
            gpu: &self.gpu,
            kv: &mut self.kv,
            cache: &self.cache,
            has_ssm,
        };
        self.live.prefetch(&mut env, tokens, 0, 0).unwrap()
    }

    /// The GPU blocks a fresh lookup matches (references returned at once) and their bytes.
    pub(super) fn gpu_hit(&mut self, tokens: &[u32]) -> Vec<Vec<u8>> {
        let m = self.cache.lookup(tokens, BS, 0, 0);
        let out = m.matched_blocks.iter().map(|&b| self.read(b)).collect();
        if m.matched_tokens > 0 {
            self.cache.release_matched(tokens, BS, m.matched_tokens, 0);
        }
        out
    }

    pub(super) fn held(&self) -> usize {
        self.live.books.lock().held.len()
    }
}

/// `n` full blocks plus one token (the last prompt token is always computed).
pub(super) fn prompt(n: usize, base: u32) -> Vec<u32> {
    (0..(n * BS + 1) as u32).map(|t| base + t).collect()
}

#[test]
fn write_through_eviction_and_restore_bring_back_identical_bytes() {
    let mut rig = Rig::eager();
    let tokens = prompt(5, 1000);
    let (table, bytes) = rig.prefill(&tokens, true);
    assert_eq!(rig.held(), 5, "one reference per started copy");
    for &b in &table {
        assert_eq!(
            rig.kv.ref_count(b),
            2,
            "the cache's reference plus the glue's"
        );
    }
    rig.reap();
    assert_eq!(rig.held(), 0, "the copies finished");
    for &b in &table {
        assert_eq!(rig.kv.ref_count(b), 1, "only the cache's reference is left");
    }
    let free_before = rig.kv.num_free_blocks();
    rig.evict_all();
    assert_eq!(rig.cache.peek_matched_tokens(&tokens, BS, 0), 0);
    assert_eq!(rig.kv.num_free_blocks(), free_before + 5);
    rig.scribble_free();

    let p = rig.prefetch(&tokens, false);
    assert_eq!(p.outcome, PrefetchOutcome::Restored);
    assert_eq!(p.gpu_tokens, 0);
    assert_eq!((p.restored_blocks, p.restored_tokens), (5, 5 * BS));
    assert_eq!(rig.cache.peek_matched_tokens(&tokens, BS, 0), 5 * BS);
    assert_eq!(
        rig.gpu_hit(&tokens),
        bytes,
        "restored blocks hold the stored bytes"
    );
    // Exactly the cache's reference on each restored block: nothing leaked, nothing lost.
    assert_eq!(rig.kv.num_free_blocks(), free_before);
    // A second prefetch finds the whole prompt on the GPU.
    assert_eq!(
        rig.prefetch(&tokens, false).outcome,
        PrefetchOutcome::GpuHit
    );
}

#[test]
fn a_partial_gpu_match_restores_only_the_missing_tail() {
    let mut rig = Rig::eager();
    let tokens = prompt(5, 2000);
    let (_, bytes) = rig.prefill(&tokens, true);
    rig.reap();
    // Leaves go first: evict until the last two blocks left the GPU and the first three stay.
    while rig.cache.peek_matched_tokens(&tokens, BS, 0) > 3 * BS {
        let ev = rig.cache.evict(1);
        assert!(!ev.is_empty(), "the tail is evictable");
        apply_evicted_blocks(ev, &mut rig.kv);
    }
    assert_eq!(rig.cache.peek_matched_tokens(&tokens, BS, 0), 3 * BS);
    rig.scribble_free();
    let p = rig.prefetch(&tokens, false);
    assert_eq!(p.outcome, PrefetchOutcome::Restored);
    assert_eq!((p.gpu_tokens, p.restored_blocks), (3 * BS, 2));
    assert_eq!(rig.gpu_hit(&tokens), bytes);
}

#[test]
fn a_d2h_in_flight_keeps_its_block_through_eviction() {
    let mut rig = Rig::eager();
    rig.gpu.set_unique_events(true);
    rig.gpu.hold_events(true);
    let tokens = prompt(4, 3000);
    let (table, _) = rig.prefill(&tokens, true);
    rig.reap();
    assert_eq!(rig.held(), 4, "copies still in flight");
    // Not visible while staging: the tier answers a miss.
    rig.evict_all();
    for &b in &table {
        assert_eq!(
            rig.kv.ref_count(b),
            1,
            "evicted, but the copy still holds block {b}"
        );
    }
    let mut grabbed = Vec::new();
    while let Some(b) = rig.kv.try_alloc_block() {
        assert!(!table.contains(&b), "block {b} recycled while its D2H runs");
        grabbed.push(b);
    }
    for b in grabbed {
        rig.kv.dec_ref(b);
    }
    assert_eq!(rig.prefetch(&tokens, false).outcome, PrefetchOutcome::Miss);
    rig.gpu.complete_all_events();
    let free_before = rig.kv.num_free_blocks();
    rig.reap();
    assert_eq!(rig.held(), 0);
    assert_eq!(
        rig.kv.num_free_blocks(),
        free_before + 4,
        "freed once the copy is done"
    );
}

#[test]
fn a_hybrid_model_without_a_snapshot_loads_nothing() {
    let mut rig = Rig::eager();
    let tokens = prompt(5, 4000);
    rig.prefill(&tokens, true);
    rig.reap();
    rig.evict_all();
    let free = rig.kv.num_free_blocks();
    let p = rig.prefetch(&tokens, true);
    assert_eq!(p.outcome, PrefetchOutcome::NoSnapshot);
    assert_eq!(rig.kv.num_free_blocks(), free, "nothing allocated");
    assert_eq!(rig.cache.peek_matched_tokens(&tokens, BS, 0), 0);
}

#[test]
fn a_hybrid_model_restores_up_to_its_deepest_snapshot() {
    let mut rig = Rig::eager();
    let tokens = prompt(5, 5000);
    let (table, bytes) = rig.prefill(&tokens, true);
    rig.reap();
    // An intermediate SSM snapshot at block 3 (48 tokens); the index outlives KV eviction.
    rig.cache.insert_intermediate_snapshot(
        &tokens[..3 * BS],
        &table[..3],
        &[],
        BS,
        7,
        0,
        3 * BS,
        0,
    );
    rig.evict_all();
    rig.scribble_free();
    let p = rig.prefetch(&tokens, true);
    assert_eq!(p.outcome, PrefetchOutcome::Restored);
    assert_eq!(
        p.restored_blocks, 3,
        "the restore ends at the snapshot, not past it"
    );
    assert_eq!(rig.gpu_hit(&tokens), bytes[..3].to_vec());
}

#[test]
fn a_span_below_min_tokens_is_not_restored() {
    // A prompt no longer than one block has no usable block: the last token is computed.
    let mut rig = Rig::eager();
    let short: Vec<u32> = (0..BS as u32).collect();
    assert_eq!(rig.prefetch(&short, false).outcome, PrefetchOutcome::Miss);
    // --kv-tier-min-tokens 32: one block (16 tokens) is too short, two blocks qualify.
    let mut rig = Rig::with(64, 256, 2 * BS as u32, TierOptions::default());
    let one = prompt(1, 6000);
    let two = prompt(2, 6500);
    rig.prefill(&one, true);
    rig.prefill(&two, true);
    rig.reap();
    rig.evict_all();
    let free = rig.kv.num_free_blocks();
    assert_eq!(
        rig.prefetch(&one, false).outcome,
        PrefetchOutcome::BelowMinTokens
    );
    assert_eq!(rig.kv.num_free_blocks(), free);
    assert_eq!(rig.prefetch(&two, false).outcome, PrefetchOutcome::Restored);
}

#[test]
fn no_free_blocks_restores_nothing_and_leaks_nothing() {
    let mut rig = Rig::new(7, 64, TierOptions::default());
    let tokens = prompt(5, 7000);
    rig.prefill(&tokens, true);
    rig.reap();
    rig.evict_all();
    // A live sequence holds four of the seven blocks: three are free, five are needed.
    let live_seq: Vec<u32> = (0..4).map(|_| rig.kv.alloc_block().unwrap()).collect();
    let free = rig.kv.num_free_blocks();
    assert_eq!(free, 3);
    assert_eq!(
        rig.prefetch(&tokens, false).outcome,
        PrefetchOutcome::NoBlocks
    );
    assert_eq!(
        rig.kv.num_free_blocks(),
        free,
        "every allocated block went back"
    );
    for b in live_seq {
        assert_eq!(rig.kv.ref_count(b), 1);
    }
}

#[test]
fn another_namespace_misses() {
    let mut rig = Rig::eager();
    let tokens = prompt(4, 8000);
    rig.prefill(&tokens, true);
    rig.reap();
    rig.evict_all();
    // Same tier, another model fingerprint: every key differs.
    let other = Live::new(rig.live.tier.clone(), 0x0dd, BS, None);
    let mut env = Env {
        gpu: &rig.gpu,
        kv: &mut rig.kv,
        cache: &rig.cache,
        has_ssm: false,
    };
    let p = other.prefetch(&mut env, &tokens, 0, 0).unwrap();
    assert_eq!(p.outcome, PrefetchOutcome::Miss);
    // Different tokens in the same namespace miss as well.
    assert_eq!(
        rig.prefetch(&prompt(4, 9000), false).outcome,
        PrefetchOutcome::Miss
    );
    assert_eq!(
        rig.prefetch(&tokens, false).outcome,
        PrefetchOutcome::Restored
    );
}

#[test]
fn only_blocks_the_insert_created_are_stored() {
    // Table positions past the prompt's full blocks, and blocks the insert did not create
    // (an existing node keeps its own block), are skipped.
    assert_eq!(
        new_block_positions(5 * BS + 3, BS, &[9, 8, 7, 6, 5, 4], &[8, 6, 4]),
        vec![1, 3]
    );
    assert_eq!(
        new_block_positions(2 * BS, BS, &[1, 2, 3], &[1, 2, 3]),
        vec![0, 1]
    );
    assert!(new_block_positions(BS - 1, BS, &[1], &[1]).is_empty());
    assert!(new_block_positions(4 * BS, 0, &[1], &[1]).is_empty());
}

#[test]
fn the_namespace_separates_models_and_geometries() {
    let d = vec!["Bf16".to_string(), "Bf16".to_string()];
    let base = namespace_from(1, 16, &[512, 512, 512, 512], &d);
    assert_eq!(
        base,
        namespace_from(1, 16, &[512, 512, 512, 512], &d),
        "deterministic"
    );
    assert_ne!(
        base,
        namespace_from(2, 16, &[512, 512, 512, 512], &d),
        "model"
    );
    assert_ne!(
        base,
        namespace_from(1, 32, &[512, 512, 512, 512], &d),
        "block size"
    );
    assert_ne!(
        base,
        namespace_from(1, 16, &[256, 256, 256, 256], &d),
        "region bytes"
    );
    let fp8 = vec!["Fp8".to_string(), "Fp8".to_string()];
    assert_ne!(
        base,
        namespace_from(1, 16, &[512, 512, 512, 512], &fp8),
        "KV dtype"
    );
}
