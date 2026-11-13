// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the host tier's per-model slot ([`TierState`], the `kv_tier` field of
//! `ModelPortState`), what a serving tier holds ([`Live`]), how it is built from the
//! configuration `port_apply` published, the tier namespace (the model fingerprint folded with
//! the KV block geometry), and the teardown that drops the tier and its pinned pool while the
//! model's backend is still alive.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use avarok_core::config::ModelConfig;
use parking_lot::Mutex;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::kv_host_tier::{self, BlockKey, KvHostTier};
use spark_runtime::pinned_pool::{self, PinnedPool};

use super::pool_gpu::PoolGpu;
use super::restore::Pending;

/// One GPU block the glue holds an extra reference on, for the tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Held {
    pub(crate) block: u32,
    /// Buffered by the lazy store and not copying yet.
    pub(crate) lazy: bool,
    /// Store order: the lazy drain ranks buffered blocks by age.
    pub(crate) order: u64,
}

/// Mutable bookkeeping of a serving tier (one lock; always taken after the KV cache lock).
#[derive(Default)]
pub(crate) struct Books {
    /// Source blocks of write-backs in flight (eager) or buffered (lazy), by key. Each holds
    /// one block reference, dropped when the key comes back from `poll_writebacks` (or, for a
    /// lazy entry the tier no longer buffers, at the next drain).
    pub(crate) held: HashMap<BlockKey, Held>,
    pub(crate) next_order: u64,
    /// Async restores by ticket id (lane L27's parked requests).
    pub(crate) pending: HashMap<u64, Pending>,
    /// Free KV blocks at the previous tick (the lazy drain's demand signal).
    pub(crate) last_free: Option<usize>,
    /// Smoothed GPU blocks consumed per tick.
    pub(crate) demand: f64,
    warned: bool,
}

/// A serving tier: lane L26a's [`KvHostTier`] plus the glue's bookkeeping.
pub(crate) struct Live {
    pub(crate) tier: Arc<KvHostTier>,
    /// The namespace folded into every [`BlockKey`] (see [`namespace`]).
    pub(crate) ns: u64,
    pub(crate) block_size: usize,
    pub(crate) books: Mutex<Books>,
    pool_gpu: Option<Arc<PoolGpu>>,
}

impl Live {
    pub(crate) fn new(
        tier: Arc<KvHostTier>,
        ns: u64,
        block_size: usize,
        pool_gpu: Option<Arc<PoolGpu>>,
    ) -> Self {
        Self {
            tier,
            ns,
            block_size,
            books: Mutex::new(Books::default()),
            pool_gpu,
        }
    }

    /// The first failure of a kind is a warning, the rest are debug lines (a broken tier must
    /// not flood the log once per prefill).
    pub(crate) fn warn_once(&self, msg: &str) {
        let mut b = self.books.lock();
        if b.warned {
            tracing::debug!("{msg}");
        } else {
            b.warned = true;
            tracing::warn!("{msg}");
        }
    }

    /// Next store-order stamp.
    pub(crate) fn stamp(b: &mut Books) -> u64 {
        b.next_order += 1;
        b.next_order
    }
}

enum Slot {
    /// No hook has run with the row on yet.
    Unbuilt,
    /// The tier will not serve this model (prefix cache off, no attention KV, build failed).
    Refused,
    Live(Arc<Live>),
    /// Torn down; never rebuilt.
    Down,
}

/// The host KV tier's per-model state (`ModelPortState::kv_tier`). Inert until the
/// `kv_host_tier` row is on: its `Default` is an unbuilt slot that no row-off path touches.
pub struct TierState {
    slot: Mutex<Slot>,
}

impl Default for TierState {
    fn default() -> Self {
        Self {
            slot: Mutex::new(Slot::Unbuilt),
        }
    }
}

impl TierState {
    /// The serving tier, running `build` on the first call. A build that returns `Ok(None)` or
    /// fails is not retried.
    pub(crate) fn get_or_build(
        &self,
        build: impl FnOnce() -> Result<Option<Live>>,
    ) -> Option<Arc<Live>> {
        let mut s = self.slot.lock();
        match &*s {
            Slot::Live(l) => return Some(l.clone()),
            Slot::Refused | Slot::Down => return None,
            Slot::Unbuilt => {}
        }
        match build() {
            Ok(Some(l)) => {
                let l = Arc::new(l);
                *s = Slot::Live(l.clone());
                Some(l)
            }
            Ok(None) => {
                *s = Slot::Refused;
                None
            }
            Err(e) => {
                tracing::warn!("kv_host_tier: the host KV tier stays off for this model: {e:#}");
                *s = Slot::Refused;
                None
            }
        }
    }

    /// The serving tier if it was built (never builds).
    pub(crate) fn live(&self) -> Option<Arc<Live>> {
        match &*self.slot.lock() {
            Slot::Live(l) => Some(l.clone()),
            _ => None,
        }
    }

    /// The tier was refused or torn down for this model.
    pub(crate) fn refused(&self) -> bool {
        matches!(&*self.slot.lock(), Slot::Refused | Slot::Down)
    }

    /// Quiesce the tier (wait for its copy stream, finish every copy), drop every block
    /// reference the glue holds, free the blocks of unfinished restores, then drop the tier and
    /// its pinned pool and detach the pool's backend handle. Idempotent.
    pub(crate) fn teardown(&self, gpu: &dyn GpuBackend, kv: &mut PagedKvCache) {
        let live = match std::mem::replace(&mut *self.slot.lock(), Slot::Down) {
            Slot::Live(l) => l,
            _ => return,
        };
        // Every finished or abandoned write-back is among the held keys, so the returned keys
        // need no separate pass: every held reference is dropped below.
        let _finished = live.tier.shutdown(gpu);
        {
            let mut b = live.books.lock();
            for (_, h) in b.held.drain() {
                kv.dec_ref(h.block);
            }
            for (_, p) in b.pending.drain() {
                p.free_dst(kv);
            }
        }
        let pool_gpu = live.pool_gpu.clone();
        match Arc::try_unwrap(live) {
            Ok(l) => drop(l),
            Err(l) => {
                tracing::warn!(
                    "kv_host_tier teardown: {} other handle(s) still hold the tier; its pinned \
                     pool is released with the last one",
                    Arc::strong_count(&l) - 1
                );
            }
        }
        if let Some(p) = pool_gpu {
            p.detach();
        }
    }
}

/// Build the serving tier from the published configuration, or `Ok(None)` when it must not
/// serve this model. `gpu` must be the model's own backend (`'static` trait object): the pinned
/// pool reaches it through a [`PoolGpu`] handle that teardown detaches.
pub(crate) fn build(
    gpu: &(dyn GpuBackend + 'static),
    cfg: &ModelConfig,
    kv: &PagedKvCache,
    cache_active: bool,
) -> Result<Option<Live>> {
    let Some(tc) = kv_host_tier::config::active() else {
        return Ok(None);
    };
    if !cache_active {
        tracing::info!(
            "kv_host_tier: the prefix cache is off for this model, so the host KV tier stays off"
        );
        return Ok(None);
    }
    let first = kv
        .checked_block_regions(0)
        .context("reading the KV block layout")?;
    let block_bytes = first.bytes();
    if block_bytes == 0 {
        tracing::info!("kv_host_tier: this model has no attention KV; the host KV tier stays off");
        return Ok(None);
    }
    let budget = usize::try_from(tc.budget_bytes)
        .context("the host tier budget does not fit the address space")?;
    let chunk = pinned_pool::default_chunk_bytes().max(1);
    let pool_gpu = PoolGpu::attach(gpu);
    let handle: Arc<dyn GpuBackend> = pool_gpu.clone();
    let pool = PinnedPool::new(handle, budget, chunk.min(budget), chunk)
        .context("reserving the host tier's pinned pool")?;
    let copy_stream = gpu
        .create_copy_stream()
        .context("creating the host tier's copy stream")?;
    let tier = Arc::new(KvHostTier::new(tc, pool, block_bytes, copy_stream));
    let regions: Vec<usize> = first.regions.iter().map(|&(_, n)| n).collect();
    let ns = namespace(cfg, kv, &regions);
    tier.set_namespace(ns);
    // Lane L27's disk tier (`--kv-tier-dir`, row `kv_tier_disk`): must follow `set_namespace`,
    // because attaching adopts the records of the namespace just named. Inert with the row off.
    match spark_runtime::kv_host_tier::disk::attach_configured(&tier) {
        Ok(Some(d)) => tracing::info!(
            "kv_host_tier: disk tier attached at {} ({} GiB)",
            d.dir().display(),
            d.budget_bytes() >> 30
        ),
        Ok(None) => {}
        Err(e) => tracing::warn!("kv_host_tier: disk tier not attached (host tier only): {e:#}"),
    }
    tracing::info!(
        "kv_host_tier: serving {:.2} GiB of host RAM ({} B blocks of {} tokens, policy {}, \
         store {}, min {} tokens, namespace {ns:#018x})",
        tc.budget_bytes as f64 / (1u64 << 30) as f64,
        block_bytes,
        kv.block_size(),
        tc.policy,
        tier.options().store_mode,
        tc.min_tokens,
    );
    Ok(Some(Live::new(tier, ns, kv.block_size(), Some(pool_gpu))))
}

/// The tier namespace of this model: its config fingerprint (`ssm_tier/fingerprint.rs`, KV
/// convention `blob_bytes = 0`, `AVAROK_MODEL_ID` folded in) folded with the block geometry and
/// the per-layer KV dtypes, so a block stored by another model, another block size or another
/// KV dtype can never be a hit.
pub(crate) fn namespace(cfg: &ModelConfig, kv: &PagedKvCache, regions: &[usize]) -> u64 {
    let model_id = std::env::var("AVAROK_MODEL_ID").unwrap_or_default();
    let fp = match crate::model::ssm_tier::ModelFingerprint::derive_with_id(cfg, 0, &model_id) {
        Ok(f) => f.get(),
        Err(e) => {
            tracing::warn!("kv_host_tier: no model fingerprint ({e:#}); geometry-only namespace");
            0
        }
    };
    let dtypes: Vec<String> = (0..regions.len() / 2)
        .map(|i| format!("{:?}", kv.dtype_for_layer(i)))
        .collect();
    namespace_from(fp, kv.block_size(), regions, &dtypes)
}

/// FNV-1a over a domain tag, the fingerprint, the block size, every region size and every
/// layer dtype (each length-prefixed, so no two inputs share an encoding).
pub(crate) fn namespace_from(
    fp: u64,
    block_size: usize,
    regions: &[usize],
    dtypes: &[String],
) -> u64 {
    const DOMAIN: &[u8] = b"atlas.kv_tier_glue.namespace.v1";
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(DOMAIN);
    eat(&fp.to_le_bytes());
    eat(&(block_size as u64).to_le_bytes());
    eat(&(regions.len() as u64).to_le_bytes());
    for &r in regions {
        eat(&(r as u64).to_le_bytes());
    }
    for d in dtypes {
        eat(&(d.len() as u64).to_le_bytes());
        eat(d.as_bytes());
    }
    h
}
