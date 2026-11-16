// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: size the Quest page summaries and charge them to the
//! KV-block budget (port lane L30), at the one point of the model build that
//! knows both the KV geometry and the budget (`factory/build.rs`, right after
//! the MTP propose-pool charge, by one fragment line), and publish the layout
//! the selector allocates later.
//!
//! The summaries are keyed by physical KV block, so the arena needs one slot
//! per block the pool can hold. The block count is not known until the budget
//! is final, and the budget shrinks by this charge, so the charge uses the
//! pre-charge block count (an upper bound of the final one, the same two-pass
//! rule the MTP pool charge uses), clamped to the blocks the engine can reach
//! when the build itself clamps the pool (prefix caching inactive and
//! `AVAROK_KV_POOL_UNCLAMPED` unset). Over-reservation is at most the summary
//! share of the charge itself (about 6% of a BF16 pool on the 4B).
//!
//! With the row off (no selector installed) [`charge_summaries`] returns the
//! budget unchanged and publishes nothing: byte-identical sizing on every
//! target.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use spark_runtime::kv_cache::page_summary::{LayerGeom, SummaryLayout};
use spark_runtime::kv_cache::{KvCacheConfig, PagedKvCache};

/// What bounds the pool's block count besides the budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolBound {
    pub max_seq_len: usize,
    pub max_batch_size: usize,
    /// The prefix cache is active (the build then keeps every budget block).
    pub prefix_cache_active: bool,
    /// `--high-speed-swap` sized the pool (the seam refuses that serve; no
    /// summary is charged).
    pub high_speed_swap: bool,
}

/// The summary layout of the model being served, with a generation number
/// that changes at every model build so the selector reallocates.
#[derive(Debug)]
pub struct Geometry {
    pub generation: u64,
    pub layout: SummaryLayout,
}

static GEOMETRY: Mutex<Option<Arc<Geometry>>> = Mutex::new(None);
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// The published layout, `None` before a charge (or with the row off).
pub fn geometry() -> Option<Arc<Geometry>> {
    GEOMETRY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Publish a layout (the charge does this; tests and the microtest call it
/// directly). Returns the new generation.
pub fn publish(layout: SummaryLayout) -> u64 {
    let generation = GENERATION.fetch_add(1, Ordering::Relaxed) + 1;
    *GEOMETRY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) =
        Some(Arc::new(Geometry { generation, layout }));
    generation
}

/// The summary arena for this KV geometry and budget, and the bytes to charge.
/// `None` when no summary applies (`--high-speed-swap`).
pub fn plan(
    cfg: &KvCacheConfig,
    kv_budget: usize,
    bound: PoolBound,
    pool_unclamped: bool,
) -> Result<Option<SummaryLayout>> {
    if bound.high_speed_swap {
        return Ok(None);
    }
    let layers: Vec<LayerGeom> = (0..cfg.num_layers)
        .map(|i| {
            let (nkv, hd) = cfg.dims_for_layer(i);
            LayerGeom {
                kv_heads: u32::try_from(nkv).unwrap_or(u32::MAX),
                head_dim: u32::try_from(hd).unwrap_or(u32::MAX),
            }
        })
        .collect();
    let budget_blocks = PagedKvCache::compute_num_blocks(cfg, kv_budget).unwrap_or(0);
    let cap = if bound.prefix_cache_active || pool_unclamped {
        budget_blocks
    } else {
        let per_seq = bound.max_seq_len.div_ceil(cfg.block_size.max(1));
        let reachable = bound
            .max_batch_size
            .saturating_mul(per_seq)
            .saturating_add(bound.max_batch_size)
            .saturating_add(1);
        budget_blocks.min(reachable)
    };
    let cap = u32::try_from(cap).unwrap_or(u32::MAX);
    let block_size = u32::try_from(cfg.block_size).unwrap_or(u32::MAX);
    SummaryLayout::new(cap, block_size, layers).map(Some)
}

/// Charge the Quest summaries to the KV budget and publish their layout.
/// Returns the budget unchanged when the selector is not installed.
pub fn charge_summaries(cfg: &KvCacheConfig, kv_budget: usize, bound: PoolBound) -> Result<usize> {
    if super::installed().is_none() {
        return Ok(kv_budget);
    }
    let unclamped = std::env::var("AVAROK_KV_POOL_UNCLAMPED").is_ok();
    let Some(layout) = plan(cfg, kv_budget, bound, unclamped)? else {
        tracing::warn!(
            "attn_page_sparse: no page summaries for a --high-speed-swap pool; attention stays dense"
        );
        return Ok(kv_budget);
    };
    let charge = usize::try_from(layout.total_bytes()).unwrap_or(usize::MAX);
    ensure!(
        charge < kv_budget,
        "attn_page_sparse: the Quest page summaries need {:.2} GiB ({} blocks), more than the whole \
         KV budget ({:.2} GiB); lower --max-seq-len or --max-batch-size, or raise \
         --gpu-memory-utilization",
        charge as f64 / f64::from(1u32 << 30),
        layout.cap_blocks,
        kv_budget as f64 / f64::from(1u32 << 30),
    );
    tracing::info!(
        "KV budget: reserving {:.1} MiB for the Quest page summaries (attn_page_sparse: {} block \
         slots x {} B, {}-token pages over {} attention layers)",
        charge as f64 / f64::from(1u32 << 20),
        layout.cap_blocks,
        SummaryLayout::bytes_per_block(&layout.layers),
        layout.page_tokens(),
        layout.layers.len(),
    );
    publish(layout);
    Ok(kv_budget - charge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::kv_cache::KvCacheDtype;

    fn cfg_4b(dtype: KvCacheDtype) -> KvCacheConfig {
        KvCacheConfig {
            block_size: 16,
            num_kv_heads: 4,
            head_dim: 256,
            num_layers: 8,
            dtype,
            layer_dtypes: Vec::new(),
            layer_dims: Vec::new(),
            cache_blocks_per_seq: None,
        }
    }

    fn bound(prefix: bool) -> PoolBound {
        PoolBound {
            max_seq_len: 131_072,
            max_batch_size: 1,
            prefix_cache_active: prefix,
            high_speed_swap: false,
        }
    }

    #[test]
    fn a_clamped_pool_charges_its_reachable_blocks_only() {
        let cfg = cfg_4b(KvCacheDtype::Bf16);
        // 8 GiB of BF16 budget is 16,384 blocks, far more than one 128K sequence reaches.
        let l = plan(&cfg, 8 << 30, bound(false), false).unwrap().unwrap();
        assert_eq!(l.cap_blocks, 8192 + 1 + 1);
        // About 256 MiB for 128K tokens on the 4B: 8,194 slots x 33,792 B.
        assert_eq!(l.total_bytes() / (1 << 20), 8194 * 33_792 / (1 << 20));
    }

    #[test]
    fn an_active_prefix_cache_charges_every_budget_block() {
        let cfg = cfg_4b(KvCacheDtype::Fp8);
        let budget = 4usize << 30;
        let l = plan(&cfg, budget, bound(true), false).unwrap().unwrap();
        let blocks = PagedKvCache::compute_num_blocks(&cfg, budget).unwrap();
        assert_eq!(l.cap_blocks as usize, blocks);
        // The same with the build's own kill switch.
        let l2 = plan(&cfg, budget, bound(false), true).unwrap().unwrap();
        assert_eq!(l2.cap_blocks as usize, blocks);
    }

    #[test]
    fn the_charge_bounds_the_final_pool() {
        // The pool sized from the charged budget never has more blocks than slots.
        let cfg = cfg_4b(KvCacheDtype::Bf16);
        for budget in [1usize << 30, 3 << 30, 7 << 30] {
            let l = plan(&cfg, budget, bound(true), false).unwrap().unwrap();
            let left = budget - l.total_bytes() as usize;
            let final_blocks = PagedKvCache::compute_num_blocks(&cfg, left).unwrap();
            assert!(final_blocks <= l.cap_blocks as usize, "{budget}");
        }
    }

    #[test]
    fn high_speed_swap_charges_nothing() {
        let cfg = cfg_4b(KvCacheDtype::Bf16);
        let b = PoolBound {
            high_speed_swap: true,
            ..bound(false)
        };
        assert!(plan(&cfg, 1 << 30, b, false).unwrap().is_none());
    }

    #[test]
    fn per_layer_dims_are_kept() {
        let mut cfg = cfg_4b(KvCacheDtype::Bf16);
        cfg.num_layers = 2;
        cfg.layer_dims = vec![(4, 256), (2, 128)];
        let l = plan(&cfg, 1 << 30, bound(true), false).unwrap().unwrap();
        assert_eq!(
            l.layers[1],
            LayerGeom {
                kv_heads: 2,
                head_dim: 128
            }
        );
    }

    #[test]
    fn without_a_selector_the_budget_is_unchanged() {
        // Unit tests never install the selector (that would register it with the
        // process-wide seam), so this is the row-off path every target takes.
        if super::super::installed().is_none() {
            let cfg = cfg_4b(KvCacheDtype::Bf16);
            assert_eq!(
                charge_summaries(&cfg, 123_456_789, bound(false)).unwrap(),
                123_456_789
            );
        }
    }
}
