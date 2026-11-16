// SPDX-License-Identifier: AGPL-3.0-only

//! CPU oracle for the copy-on-write prefix tail (`prefix_cow.rs`, row
//! `prefix_cow_tail`, issue #1193). The mock backend holds device memory as
//! host buffers, so the tests check the copied BYTES, the reference counts,
//! the stream every copy runs on, and the drain on the last-holder path.

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::prefix_cache::{NoPrefixCaching, PrefixCache};
use spark_runtime::radix_tree::RadixTree;

use super::{copy_plan, copy_tail_block, cow_lookup_allowed};

const BS: usize = 16;
const STREAM: u64 = 0x5eed;

/// Three attention layers with different dtypes and dims, so K and V strides
/// differ per layer and between sides (the asymmetric layer).
fn mixed_config() -> KvCacheConfig {
    KvCacheConfig {
        block_size: BS,
        num_kv_heads: 2,
        head_dim: 64,
        num_layers: 3,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![
            KvCacheDtype::Bf16,
            KvCacheDtype::Fp8,
            KvCacheDtype::Bf16KTurbo4V,
        ],
        layer_dims: vec![(2, 64), (4, 32), (2, 64)],
        cache_blocks_per_seq: None,
    }
}

/// Fill every layer's K and V of block `b` with a pattern unique to
/// `(layer, side, block, tag)`.
fn fill(gpu: &MockGpuBackend, kv: &PagedKvCache, b: u32, tag: u8) {
    for l in 0..kv.num_layers() {
        let k =
            vec![tag ^ (l as u8) ^ (b as u8).wrapping_mul(3); kv.k_block_stride_bytes_for_layer(l)];
        let v = vec![
            !tag ^ (l as u8) ^ (b as u8).wrapping_mul(5);
            kv.v_block_stride_bytes_for_layer(l)
        ];
        gpu.copy_h2d(&k, kv.k_cache_ptr(l, b)).unwrap();
        gpu.copy_h2d(&v, kv.v_cache_ptr(l, b)).unwrap();
    }
}

/// Block `b`'s bytes, every layer, K then V.
fn read(gpu: &MockGpuBackend, kv: &PagedKvCache, b: u32) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for l in 0..kv.num_layers() {
        let mut k = vec![0u8; kv.k_block_stride_bytes_for_layer(l)];
        let mut v = vec![0u8; kv.v_block_stride_bytes_for_layer(l)];
        gpu.copy_d2h(kv.k_cache_ptr(l, b), &mut k).unwrap();
        gpu.copy_d2h(kv.v_cache_ptr(l, b), &mut v).unwrap();
        out.push(k);
        out.push(v);
    }
    out
}

#[test]
fn the_plan_covers_every_attention_layer_k_and_v_exactly_once() {
    let gpu = MockGpuBackend::new();
    let kv = PagedKvCache::new(mixed_config(), 6, &gpu).unwrap();
    let plan = copy_plan(&kv, 2, 5);
    assert_eq!(plan.len(), 2 * kv.num_layers());
    let mut seen = std::collections::HashSet::new();
    for c in &plan {
        assert!(
            seen.insert((c.layer, c.v)),
            "{:?} copied twice",
            (c.layer, c.v)
        );
        let (src, dst, bytes) = if c.v {
            (
                kv.v_cache_ptr(c.layer, 2),
                kv.v_cache_ptr(c.layer, 5),
                kv.v_block_stride_bytes_for_layer(c.layer),
            )
        } else {
            (
                kv.k_cache_ptr(c.layer, 2),
                kv.k_cache_ptr(c.layer, 5),
                kv.k_block_stride_bytes_for_layer(c.layer),
            )
        };
        assert_eq!((c.src, c.dst, c.bytes), (src, dst, bytes));
    }
    assert_ne!(
        kv.k_block_stride_bytes_for_layer(2),
        kv.v_block_stride_bytes_for_layer(2),
        "the asymmetric layer must exercise distinct K and V strides"
    );
}

#[test]
fn the_private_block_holds_the_source_bytes_and_later_donor_writes_do_not_leak() {
    let gpu = MockGpuBackend::new();
    let mut kv = PagedKvCache::new(mixed_config(), 6, &gpu).unwrap();
    let src = kv.try_alloc_block().unwrap(); // the cache's (or donor's) reference
    let other = kv.try_alloc_block().unwrap();
    fill(&gpu, &kv, src, 0x11);
    fill(&gpu, &kv, other, 0x77);
    let other_before = read(&gpu, &kv, other);
    let d2d0 = gpu.d2d_count();

    let dst = copy_tail_block(&gpu, &mut kv, &NoPrefixCaching, src, STREAM)
        .unwrap()
        .expect("a free block exists");
    assert_ne!(dst, src);
    assert_eq!(
        read(&gpu, &kv, dst),
        read(&gpu, &kv, src),
        "byte-exact copy"
    );
    assert_eq!(
        read(&gpu, &kv, other),
        other_before,
        "no other block touched"
    );
    assert_eq!(gpu.d2d_count() - d2d0, 2 * kv.num_layers());
    assert!(
        gpu.d2d_async_streams().iter().all(|&s| s == STREAM),
        "prefill stream only"
    );
    assert_eq!(
        (kv.ref_count(src), kv.ref_count(dst)),
        (1, 1),
        "hold released"
    );

    // The donor keeps writing its block; the requester's copy is its own.
    let copied = read(&gpu, &kv, dst);
    fill(&gpu, &kv, src, 0x99);
    assert_eq!(read(&gpu, &kv, dst), copied);
}

#[test]
fn a_source_that_is_not_live_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut kv = PagedKvCache::new(mixed_config(), 4, &gpu).unwrap();
    let free_before = kv.num_free_blocks();
    let d2d0 = gpu.d2d_count();
    assert_eq!(
        copy_tail_block(&gpu, &mut kv, &NoPrefixCaching, 1, STREAM).unwrap(),
        None
    );
    assert_eq!(
        copy_tail_block(&gpu, &mut kv, &NoPrefixCaching, 99, STREAM).unwrap(),
        None
    );
    assert_eq!(kv.num_free_blocks(), free_before);
    assert_eq!(kv.ref_count(1), 0);
    assert_eq!(gpu.d2d_count(), d2d0);
}

#[test]
fn an_exhausted_pool_falls_back_with_references_unchanged() {
    let gpu = MockGpuBackend::new();
    let mut kv = PagedKvCache::new(mixed_config(), 2, &gpu).unwrap();
    let src = kv.try_alloc_block().unwrap();
    let _held = kv.try_alloc_block().unwrap();
    let d2d0 = gpu.d2d_count();
    assert_eq!(
        copy_tail_block(&gpu, &mut kv, &NoPrefixCaching, src, STREAM).unwrap(),
        None
    );
    assert_eq!(kv.ref_count(src), 1);
    assert_eq!(kv.num_free_blocks(), 0);
    assert_eq!(gpu.d2d_count(), d2d0);
}

/// The allocation evicts the source's radix node, so the copy's hold is the
/// last reference: the stream must drain before the block returns to the
/// free list, or its next owner could overwrite it under the copy.
#[test]
fn the_last_holder_drains_the_stream_before_freeing_the_source() {
    let gpu = MockGpuBackend::new();
    let mut kv = PagedKvCache::new(mixed_config(), 3, &gpu).unwrap();
    let tree = RadixTree::with_levers(true, true);
    let a = kv.try_alloc_block().unwrap(); // cached, least recently used
    let c = kv.try_alloc_block().unwrap(); // cached, newer
    let _held = kv.try_alloc_block().unwrap(); // a live sequence's block
    let ta: Vec<u32> = (0..16).collect();
    let tc: Vec<u32> = (100..116).collect();
    // matched_tokens == len: the cache holds the only ref (ref_count 1).
    tree.insert(&ta, &[a], &[], BS, BS, 0);
    tree.insert(&tc, &[c], &[], BS, BS, 0);
    fill(&gpu, &kv, a, 0x21);
    let want = read(&gpu, &kv, a);
    let syncs0 = gpu.sync_count();

    let dst = copy_tail_block(&gpu, &mut kv, &tree, a, STREAM)
        .unwrap()
        .expect("evicting both nodes frees block c");
    assert_eq!(dst, c, "a stayed held by the copy, so c is the block freed");
    assert_eq!(read(&gpu, &kv, dst), want);
    assert!(
        gpu.sync_count() > syncs0,
        "the copy must drain before a is freed"
    );
    assert_eq!(kv.ref_count(a), 0, "a is free again");
    assert_eq!(kv.num_free_blocks(), 1);
    assert_eq!(tree.stats(), (0, 0), "both nodes were evicted");
}

/// End to end through the radix tree: a CoW lookup, the copy, the sequence's
/// table, and the references after release.
#[test]
fn a_cow_lookup_and_copy_leave_balanced_references() {
    let gpu = MockGpuBackend::new();
    let mut kv = PagedKvCache::new(mixed_config(), 8, &gpu).unwrap();
    let tree = RadixTree::with_levers(false, true);
    // Donor: 20-token prompt, full block b0 plus a 4-token tail in b1, both
    // cached (the cache's reference) and still held by the donor.
    let b0 = kv.try_alloc_block().unwrap();
    let b1 = kv.try_alloc_block().unwrap();
    let donor: Vec<u32> = (0..20).collect();
    let acq = tree.insert(&donor, &[b0, b1], &[], BS, 0, 0);
    for &b in &acq.blocks {
        kv.inc_ref(b);
    }
    fill(&gpu, &kv, b1, 0x42);

    let mut m = tree.lookup_cow(&donor, BS, 0, 0);
    assert_eq!(m.matched_blocks, vec![b0]);
    assert_eq!(m.cow_tail, Some((b1, 4)));
    let (src, _) = m.cow_tail.take().unwrap();
    let mut table = Vec::new();
    for &b in &m.matched_blocks {
        kv.inc_ref(b);
        table.push(b);
    }
    let dst = copy_tail_block(&gpu, &mut kv, &tree, src, STREAM)
        .unwrap()
        .unwrap();
    table.push(dst);
    assert!(
        !table.contains(&b1),
        "the donor's tail is never in the table"
    );
    assert_eq!(read(&gpu, &kv, dst), read(&gpu, &kv, b1));
    assert_eq!(kv.ref_count(b1), 2, "donor + cache; the hold is gone");

    // The requester finishes: tree release, then its blocks.
    tree.release(&donor, BS, 0);
    kv.free_blocks(&table);
    assert_eq!(
        kv.ref_count(dst),
        0,
        "the private copy was the requester's alone"
    );
    assert_eq!(kv.ref_count(b0), 2, "donor + cache");
}

#[test]
fn cow_lookup_is_refused_in_a_multi_rank_world_and_under_high_speed_swap() {
    let gpu = MockGpuBackend::new();
    let kv = PagedKvCache::new(mixed_config(), 2, &gpu).unwrap();
    assert!(cow_lookup_allowed(false, &kv));
    assert!(!cow_lookup_allowed(true, &kv));
    let mut hss = mixed_config();
    hss.cache_blocks_per_seq = Some(4);
    let kv = PagedKvCache::new(hss, 2, &gpu).unwrap();
    assert!(!cow_lookup_allowed(false, &kv));
}

/// The trait default: a cache without CoW answers `lookup_cow` as `lookup`.
#[test]
fn a_cache_without_cow_never_offers_a_tail() {
    let cache = NoPrefixCaching;
    let m = cache.lookup_cow(&[1, 2, 3], BS, 0, 0);
    assert!(m.is_empty() && m.cow_tail.is_none());
}
