// SPDX-License-Identifier: AGPL-3.0-only

//! W1.4 Gemma-4 E2B cross-layer KV sharing tests.
//!
//! Split out of `tests.rs` for the 500-LoC cap. These pin the pool-map
//! behavior: 35 logical layers over 15 physical pools, with shared layers
//! 15-34 reading their producer's pool (sliding → 13, full → 14).

use super::*;
use crate::gpu::mock::MockGpuBackend;

fn test_config() -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 256,
        num_layers: 12,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        layer_to_pool: vec![],
        cache_blocks_per_seq: None,
    }
}

/// W1.4 Gemma-4 E2B pool map: 35 logical layers over 15 physical pools.
/// Layers 0-14 own their pool; 15-34 route to producer 13 (sliding) / 14 (full).
fn e2b_kv_pool_map() -> Vec<usize> {
    (0..35)
        .map(|i| {
            if i < 15 {
                i
            } else if i % 5 == 4 {
                14
            } else {
                13
            }
        })
        .collect()
}

/// W1.4: empty `layer_to_pool` = identity mapping — every model that never
/// enabled sharing is byte-identical to before.
#[test]
fn test_pool_for_layer_identity_when_empty() {
    let cfg = test_config(); // layer_to_pool empty
    for i in 0..12 {
        assert_eq!(cfg.pool_for_layer(i), i);
    }
    // Index beyond any map stays identity.
    assert_eq!(cfg.pool_for_layer(40), 40);
}

/// W1.4: `pool_for_layer` returns the map entry for shared layers.
#[test]
fn test_pool_for_layer_e2b_map() {
    let map = e2b_kv_pool_map();
    let cfg = KvCacheConfig {
        layer_to_pool: map.clone(),
        ..test_config()
    };
    for (i, &pool) in map.iter().enumerate() {
        assert_eq!(cfg.pool_for_layer(i), pool, "layer {i}");
    }
}

/// W1.4 Gemma-4 E2B: PagedKvCache with 15 physical pools + the shared map —
/// every per-layer accessor routes a logical layer to the producer's pool.
#[test]
fn test_e2b_shared_pool_routing() {
    let gpu = MockGpuBackend::new();
    let layer_dims: Vec<(usize, usize)> = (0..35)
        .map(|i| if i % 5 == 4 { (1, 512) } else { (1, 256) })
        .collect();
    let cfg = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 256,
        num_layers: 15,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims,
        layer_to_pool: e2b_kv_pool_map(),
        cache_blocks_per_seq: None,
    };
    let cache = PagedKvCache::new(cfg, 4, &gpu).unwrap();

    // Shared layers read the producer's pool (same pointer), producers distinct.
    assert_eq!(cache.k_pool_ptr(13), cache.k_pool_ptr(15));
    assert_eq!(cache.k_pool_ptr(14), cache.k_pool_ptr(19));
    assert_eq!(cache.v_pool_ptr(13), cache.v_pool_ptr(20));
    assert_ne!(cache.k_pool_ptr(13), cache.k_pool_ptr(14));

    // Per-block pointers route the same way.
    assert_eq!(cache.k_cache_ptr(13, 1), cache.k_cache_ptr(15, 1));
    assert_eq!(cache.v_cache_ptr(14, 0), cache.v_cache_ptr(34, 0));

    // Producer's dims (sliding 256 / full 512) are what the shared layer reads.
    assert_eq!(cache.config().dims_for_layer(15), (1, 256));
    assert_eq!(cache.config().dims_for_layer(19), (1, 512));
    assert_eq!(
        cache.config().dims_for_layer(15),
        cache.config().dims_for_layer(13)
    );
    assert_eq!(
        cache.config().dims_for_layer(19),
        cache.config().dims_for_layer(14)
    );

    // Block strides follow the producer's pool layout.
    assert_eq!(
        cache.block_stride_bytes_for_layer(15),
        cache.block_stride_bytes_for_layer(13)
    );
    assert_eq!(
        cache.k_block_stride_bytes_for_layer(19),
        cache.k_block_stride_bytes_for_layer(14)
    );
    assert_eq!(
        cache.v_block_stride_bytes_for_layer(15),
        cache.v_block_stride_bytes_for_layer(13)
    );
    assert_eq!(cache.block_stride_bytes_for_layer(13), 4096); // 16*1*256 fp8
    assert_eq!(cache.block_stride_bytes_for_layer(14), 8192); // 16*1*512 fp8
}

/// W1.4 regression: identity map (empty layer_to_pool) keeps every pool
/// distinct — no accidental aliasing for existing models.
#[test]
fn test_identity_map_pools_distinct() {
    let gpu = MockGpuBackend::new();
    let cache = PagedKvCache::new(test_config(), 4, &gpu).unwrap(); // 12 layers
    let mut ptrs = Vec::new();
    for i in 0..12 {
        let p = cache.k_pool_ptr(i);
        assert!(!ptrs.contains(&p), "layer {i} aliases an earlier pool");
        ptrs.push(p);
    }
    assert_eq!(cache.num_layers(), 12);
}
