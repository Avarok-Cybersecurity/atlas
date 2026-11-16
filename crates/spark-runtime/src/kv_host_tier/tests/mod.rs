// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the host tier's test rig (mock backend whose device memory is host buffers,
//! an in-memory host arena, a hand-driven clock, block and key helpers) and the test modules.
//! Every test builds its own tier with `register_metrics: false`, so tests never share state
//! through the process-wide `/metrics` registration.

mod arc_ref;
mod arc_trace;
mod cold;
mod config_cost;
mod lazy_store;
mod lease;
mod metrics;
mod policy_order;
mod store_lookup;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use parking_lot::Mutex;

use super::{BlockKey, Clock, HostArena, HostRun, KvHostTier, TierConfig, TierOptions, TierPolicy};
use crate::gpu::mock::MockGpuBackend;
use crate::gpu::{DevicePtr, GpuBackend};
use crate::kv_cache::block_io::BlockRegions;

/// A host arena over boxed byte buffers. `max_runs` simulates a pool that cannot place more.
pub(super) struct VecArena {
    runs: Mutex<HashMap<u64, Box<[u8]>>>,
    next: AtomicU64,
    max_runs: AtomicUsize,
}

impl VecArena {
    pub(super) fn new() -> Self {
        Self {
            runs: Mutex::new(HashMap::new()),
            next: AtomicU64::new(1),
            max_runs: AtomicUsize::new(usize::MAX),
        }
    }

    pub(super) fn live(&self) -> usize {
        self.runs.lock().len()
    }

    pub(super) fn set_max_runs(&self, n: usize) {
        self.max_runs.store(n, Ordering::Relaxed);
    }
}

impl HostArena for VecArena {
    fn alloc(&self, bytes: usize) -> Option<HostRun> {
        let mut runs = self.runs.lock();
        if runs.len() >= self.max_runs.load(Ordering::Relaxed) || bytes == 0 {
            return None;
        }
        // Filled with a poison byte: a restore that read an unwritten run would show it.
        let mut buf = vec![0xEEu8; bytes].into_boxed_slice();
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        // A mutable-provenance address: the tier and the mock copy engine write through it.
        let ptr = buf.as_mut_ptr() as usize;
        runs.insert(id, buf);
        Some(HostRun { ptr, bytes, id })
    }

    fn free(&self, run: HostRun) {
        let removed = self.runs.lock().remove(&run.id);
        assert!(
            removed.is_some(),
            "freed a run the arena does not hold: {run:?}"
        );
    }
}

/// A clock the test advances by hand.
#[derive(Default)]
pub(super) struct ManualClock(Mutex<Duration>);

impl ManualClock {
    pub(super) fn advance(&self, d: Duration) {
        *self.0.lock() += d;
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Duration {
        *self.0.lock()
    }
}

/// Layers per test block and bytes per region (K and V of each layer): 3 x 2 x 64 = 384 bytes.
pub(super) const LAYERS: usize = 3;
pub(super) const REGION: usize = 64;
pub(super) const BLOCK_BYTES: usize = LAYERS * 2 * REGION;
/// Tokens per test block.
pub(super) const BS: usize = 4;
/// Test namespace (the model fingerprint).
pub(super) const NS: u64 = 0xA71A5;

pub(super) struct Rig {
    pub(super) gpu: Arc<MockGpuBackend>,
    pub(super) arena: Arc<VecArena>,
    pub(super) clock: Arc<ManualClock>,
    pub(super) tier: KvHostTier,
}

/// Defaults for tests: no `/metrics` registration.
pub(super) fn test_opts() -> TierOptions {
    TierOptions {
        register_metrics: false,
        namespace: NS,
        ..TierOptions::default()
    }
}

/// A budget of `blocks` test blocks.
pub(super) fn cfg_blocks(blocks: u64, policy: TierPolicy) -> TierConfig {
    TierConfig {
        budget_bytes: blocks * BLOCK_BYTES as u64,
        writeback_bytes_per_step: 1 << 30,
        min_tokens: 1,
        policy,
        restore_gate_ratio: 0.5,
    }
}

pub(super) fn rig_with(cfg: TierConfig, opts: TierOptions) -> Rig {
    let gpu = Arc::new(MockGpuBackend::new());
    gpu.set_unique_events(true);
    let arena = Arc::new(VecArena::new());
    let clock = Arc::new(ManualClock::default());
    let copy_stream = gpu.create_copy_stream().expect("copy stream");
    let tier = KvHostTier::with_parts(
        cfg,
        arena.clone(),
        BLOCK_BYTES,
        copy_stream,
        opts,
        clock.clone(),
    );
    Rig {
        gpu,
        arena,
        clock,
        tier,
    }
}

pub(super) fn rig(blocks: u64, policy: TierPolicy) -> Rig {
    rig_with(cfg_blocks(blocks, policy), test_opts())
}

/// One device block (LAYERS x K,V regions) filled with bytes derived from `seed`.
pub(super) fn device_block(gpu: &MockGpuBackend, seed: u8) -> BlockRegions {
    let mut regions = Vec::with_capacity(LAYERS * 2);
    for r in 0..LAYERS * 2 {
        let ptr = gpu.alloc(REGION).expect("mock alloc");
        let bytes: Vec<u8> = (0..REGION)
            .map(|i| seed.wrapping_mul(31).wrapping_add((r * REGION + i) as u8))
            .collect();
        gpu.copy_h2d(&bytes, ptr).expect("fill");
        regions.push((ptr, REGION));
    }
    BlockRegions { regions }
}

/// A fresh, zeroed device block (a restore destination).
pub(super) fn empty_block(gpu: &MockGpuBackend) -> BlockRegions {
    let regions = (0..LAYERS * 2)
        .map(|_| (gpu.alloc(REGION).expect("mock alloc"), REGION))
        .collect();
    BlockRegions { regions }
}

/// Every byte of a block, region by region.
pub(super) fn read_block(gpu: &MockGpuBackend, b: &BlockRegions) -> Vec<u8> {
    let mut out = Vec::with_capacity(BLOCK_BYTES);
    for &(ptr, bytes) in &b.regions {
        let mut buf = vec![0u8; bytes];
        gpu.copy_d2h(ptr, &mut buf).expect("read");
        out.extend_from_slice(&buf);
    }
    out
}

/// Overwrite a device block (its GPU copy was evicted and the block reused).
pub(super) fn clobber(gpu: &MockGpuBackend, b: &BlockRegions) {
    for &(ptr, bytes) in &b.regions {
        gpu.copy_h2d(&vec![0x5Au8; bytes], ptr).expect("clobber");
    }
}

/// A prompt of `blocks` full blocks: tokens, per-block token slices, chained keys.
pub(super) struct Prompt {
    pub(super) tokens: Vec<u32>,
    pub(super) keys: Vec<BlockKey>,
}

impl Prompt {
    pub(super) fn new(ns: u64, first_token: u32, blocks: usize) -> Self {
        let tokens: Vec<u32> = (0..(blocks * BS) as u32).map(|i| first_token + i).collect();
        let keys = BlockKey::chain_blocks(ns, 0, &tokens, BS);
        Self { tokens, keys }
    }

    pub(super) fn chunks(&self) -> Vec<&[u32]> {
        self.tokens.chunks_exact(BS).collect()
    }
}

/// Store `blocks` (one per prompt block) and publish them (events complete at once).
pub(super) fn store_published(r: &Rig, p: &Prompt, blocks: &[BlockRegions]) -> usize {
    let n = r
        .tier
        .store(r.gpu.as_ref(), &p.keys, &p.chunks(), blocks)
        .expect("store");
    r.gpu.complete_all_events();
    let done = r.tier.poll_writebacks(r.gpu.as_ref());
    assert_eq!(done.len(), n, "every started key comes back once");
    n
}

/// A device pointer nothing allocated (a copy from it fails in the mock).
pub(super) fn bogus_block() -> BlockRegions {
    BlockRegions {
        regions: (0..LAYERS * 2)
            .map(|i| (DevicePtr(0x7F00_0000 + (i * REGION) as u64), REGION))
            .collect(),
    }
}
