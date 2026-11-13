// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU oracles for `kv_cache::block_io` and its `swap` child on the mock backend.
//!
//! The oracle for every image is the per-region path the swap file has always been written
//! with: `PagedKvCache::read_block` per block per layer (K then V), then the SSM regions. A
//! batched gather, a chunked stream through the staging buffer and a scatter must reproduce
//! those bytes exactly, for symmetric and asymmetric K/V dtypes, with the backend's batched
//! entry point on and off.

use std::io::Cursor;

use super::swap::{
    SwapDir, SwapReport, SwapStaging, format_report, read_image_with, render_prometheus,
    staging_half_bytes, totals, write_image_with,
};
use super::*;
use crate::gpu::mock::MockGpuBackend;
use crate::gpu::{DevicePtr, GpuBackend};
use crate::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

fn config(dtype: KvCacheDtype, layers: usize) -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 64,
        num_layers: layers,
        dtype,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

/// A deterministic byte pattern, distinct per `(seed, i)`.
fn pattern(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

/// Fill every block of `table` in every layer with a distinct pattern.
fn fill(gpu: &MockGpuBackend, kv: &PagedKvCache, table: &[u32]) {
    for &b in table {
        for l in 0..kv.num_layers() {
            let k = pattern(
                u64::from(b) * 1000 + l as u64 * 2,
                kv.k_block_stride_bytes_for_layer(l),
            );
            let v = pattern(
                u64::from(b) * 1000 + l as u64 * 2 + 1,
                kv.v_block_stride_bytes_for_layer(l),
            );
            kv.write_block(l, b, &k, &v, gpu).unwrap();
        }
    }
}

/// The legacy swap order: for each block, for each layer, K then V.
fn legacy_kv_bytes(gpu: &MockGpuBackend, kv: &PagedKvCache, table: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &b in table {
        for l in 0..kv.num_layers() {
            let (k, v) = kv.read_block(l, b, gpu).unwrap();
            out.extend_from_slice(&k);
            out.extend_from_slice(&v);
        }
    }
    out
}

fn alloc_table(kv: &mut PagedKvCache, n: usize) -> Vec<u32> {
    (0..n).map(|_| kv.alloc_block().unwrap()).collect()
}

fn flat(blocks: &[BlockRegions]) -> Vec<(DevicePtr, usize)> {
    blocks
        .iter()
        .flat_map(|b| b.regions.iter().copied())
        .collect()
}

#[test]
fn block_regions_are_k_then_v_per_layer_with_per_side_strides() {
    for dtype in [
        KvCacheDtype::Bf16,
        KvCacheDtype::Fp8,
        KvCacheDtype::Bf16KTurbo3V,
    ] {
        let gpu = MockGpuBackend::new();
        let kv = PagedKvCache::new(config(dtype, 3), 6, &gpu).unwrap();
        let r = kv.block_regions(4);
        assert_eq!(r.len(), 6, "{dtype}");
        for l in 0..3 {
            assert_eq!(
                r.regions[2 * l],
                (kv.k_cache_ptr(l, 4), kv.k_block_stride_bytes_for_layer(l))
            );
            assert_eq!(
                r.regions[2 * l + 1],
                (kv.v_cache_ptr(l, 4), kv.v_block_stride_bytes_for_layer(l))
            );
        }
        let per_block: usize = (0..3)
            .map(|l| kv.k_block_stride_bytes_for_layer(l) + kv.v_block_stride_bytes_for_layer(l))
            .sum();
        assert_eq!(r.bytes(), per_block, "{dtype}");
        assert!(!r.is_empty());
    }
}

#[test]
fn asymmetric_dtypes_really_have_two_strides() {
    // Guards the test above: if this pair ever became symmetric, the per-side stride check
    // would stop covering anything.
    let gpu = MockGpuBackend::new();
    let kv = PagedKvCache::new(config(KvCacheDtype::Bf16KTurbo3V, 1), 2, &gpu).unwrap();
    assert_ne!(
        kv.k_block_stride_bytes_for_layer(0),
        kv.v_block_stride_bytes_for_layer(0)
    );
}

#[test]
fn checked_regions_reject_an_out_of_range_block() {
    let gpu = MockGpuBackend::new();
    let kv = PagedKvCache::new(config(KvCacheDtype::Fp8, 2), 4, &gpu).unwrap();
    assert!(kv.checked_block_regions(3).is_ok());
    assert!(kv.checked_block_regions(4).is_err());
    assert!(kv.table_regions(&[0, 9]).is_err());
    assert_eq!(kv.table_regions(&[2, 0]).unwrap().len(), 2);
}

#[test]
fn gather_reproduces_the_legacy_bytes_with_batching_on_and_off() {
    for dtype in [KvCacheDtype::Bf16, KvCacheDtype::Bf16KTurbo3V] {
        for batched in [false, true] {
            let gpu = MockGpuBackend::new();
            gpu.set_copy_batch_supported(batched);
            let mut kv = PagedKvCache::new(config(dtype, 4), 12, &gpu).unwrap();
            let mut table = alloc_table(&mut kv, 7);
            table.reverse(); // a table that is not in physical order
            fill(&gpu, &kv, &table);
            let blocks = kv.table_regions(&table).unwrap();
            let mut host = vec![0u8; total_bytes(&blocks)];
            let n = gather_to_host(&gpu, &blocks, &mut host, 0).unwrap();
            gpu.synchronize(0).unwrap();
            assert_eq!(n, host.len());
            assert_eq!(
                host,
                legacy_kv_bytes(&gpu, &kv, &table),
                "{dtype} batched={batched}"
            );
            if batched {
                assert!(gpu.copy_batch_call_count() >= 1);
                assert_eq!(gpu.copy_batch_entry_count(), 7 * 4 * 2);
            } else {
                assert_eq!(gpu.copy_batch_call_count(), 0);
            }
        }
    }
}

#[test]
fn scatter_restores_the_image_into_other_blocks() {
    let gpu = MockGpuBackend::new();
    gpu.set_copy_batch_supported(true);
    let mut kv = PagedKvCache::new(config(KvCacheDtype::Fp8, 3), 16, &gpu).unwrap();
    let src = alloc_table(&mut kv, 5);
    fill(&gpu, &kv, &src);
    let want = legacy_kv_bytes(&gpu, &kv, &src);
    let dst = alloc_table(&mut kv, 5);
    let blocks = kv.table_regions(&dst).unwrap();
    scatter_from_host(&gpu, &blocks, &want, 0, true).unwrap();
    gpu.synchronize(0).unwrap();
    assert_eq!(legacy_kv_bytes(&gpu, &kv, &dst), want);
}

#[test]
fn short_host_buffers_are_refused() {
    let gpu = MockGpuBackend::new();
    let mut kv = PagedKvCache::new(config(KvCacheDtype::Fp8, 2), 4, &gpu).unwrap();
    let t = alloc_table(&mut kv, 2);
    let blocks = kv.table_regions(&t).unwrap();
    let mut short = vec![0u8; total_bytes(&blocks) - 1];
    assert!(gather_to_host(&gpu, &blocks, &mut short, 0).is_err());
    assert!(scatter_from_host(&gpu, &blocks, &short, 0, false).is_err());
}

#[test]
fn per_block_host_addresses_round_trip() {
    let gpu = MockGpuBackend::new();
    gpu.set_copy_batch_supported(true);
    let mut kv = PagedKvCache::new(config(KvCacheDtype::Bf16, 2), 10, &gpu).unwrap();
    let src = alloc_table(&mut kv, 3);
    fill(&gpu, &kv, &src);
    let blocks = kv.table_regions(&src).unwrap();
    let per = blocks[0].bytes();
    // Three separate host buffers, deliberately not adjacent.
    let mut bufs: Vec<Vec<u8>> = (0..3).map(|_| vec![0u8; per]).collect();
    let hosts: Vec<usize> = bufs.iter_mut().map(|b| b.as_mut_ptr() as usize).collect();
    // SAFETY: each buffer is `per` bytes, distinct, and untouched until the sync below.
    let n = unsafe { gather_to_hosts(&gpu, &blocks, &hosts, 0) }.unwrap();
    gpu.synchronize(0).unwrap();
    assert_eq!(n, 3 * per);
    let legacy = legacy_kv_bytes(&gpu, &kv, &src);
    assert_eq!(bufs.concat(), legacy);

    let dst = alloc_table(&mut kv, 3);
    let dst_blocks = kv.table_regions(&dst).unwrap();
    // SAFETY: the buffers are live, `per` bytes each, and unchanged until the sync below.
    unsafe { scatter_from_hosts(&gpu, &dst_blocks, &hosts, 0, true) }.unwrap();
    gpu.synchronize(0).unwrap();
    assert_eq!(legacy_kv_bytes(&gpu, &kv, &dst), legacy);
    // A count mismatch is refused before anything is enqueued.
    // SAFETY: refused before any copy is built.
    assert!(unsafe { gather_to_hosts(&gpu, &blocks, &hosts[..2], 0) }.is_err());
}

#[test]
fn large_batches_are_split_into_bounded_submits() {
    let gpu = MockGpuBackend::new();
    gpu.set_copy_batch_supported(true);
    // 1 layer x 2 sides per block: MAX_BATCH_ENTRIES / 2 + 3 blocks needs two submits.
    let blocks_n = MAX_BATCH_ENTRIES / 2 + 3;
    let mut kv = PagedKvCache::new(config(KvCacheDtype::Fp8, 1), blocks_n, &gpu).unwrap();
    let t = alloc_table(&mut kv, blocks_n);
    let blocks = kv.table_regions(&t).unwrap();
    let mut host = vec![0u8; total_bytes(&blocks)];
    gather_to_host(&gpu, &blocks, &mut host, 0).unwrap();
    assert_eq!(gpu.copy_batch_call_count(), 2);
    assert_eq!(gpu.copy_batch_entry_count(), blocks_n * 2);
}

/// Concatenate a chunk plan back into (device pointer, bytes) runs and compare with the input.
fn replay(plans: &[ChunkPlan]) -> Vec<(u64, usize)> {
    let mut out: Vec<(u64, usize)> = Vec::new();
    for p in plans {
        let mut off = 0usize;
        for &(dev, at, n) in &p.entries {
            assert_eq!(at, off, "entries fill the chunk in order without gaps");
            off += n;
            match out.last_mut() {
                Some(last) if last.0 + last.1 as u64 == dev.0 => last.1 += n,
                _ => out.push((dev.0, n)),
            }
        }
        assert_eq!(off, p.bytes);
    }
    out
}

#[test]
fn plan_chunks_cuts_exact_chunks_and_preserves_order() {
    let mut seed = 0x1234_5678_u64;
    let mut next = |m: usize| {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (seed >> 33) as usize % m
    };
    for _ in 0..200 {
        let n_regions = 1 + next(12);
        // Regions far apart so the replay never merges two regions by accident.
        let regions: Vec<(DevicePtr, usize)> = (0..n_regions)
            .map(|i| (DevicePtr(0x10_0000 * (i as u64 + 1)), next(5000)))
            .collect();
        let cap = 1 + next(3000);
        let plans = plan_chunks(&regions, cap).unwrap();
        let total: usize = regions.iter().map(|r| r.1).sum();
        assert_eq!(plans.iter().map(|p| p.bytes).sum::<usize>(), total);
        for (i, p) in plans.iter().enumerate() {
            if i + 1 < plans.len() {
                assert_eq!(p.bytes, cap, "every chunk but the last is full");
            }
            assert!(p.bytes > 0 && p.bytes <= cap);
        }
        let want: Vec<(u64, usize)> = regions
            .iter()
            .filter(|r| r.1 > 0)
            .map(|r| (r.0.0, r.1))
            .collect();
        assert_eq!(replay(&plans), want);
    }
    assert!(plan_chunks(&[(DevicePtr(64), 8)], 0).is_err());
    assert!(plan_chunks(&[], 16).unwrap().is_empty());
    assert!(plan_chunks(&[(DevicePtr(64), 0)], 16).unwrap().is_empty());
}

#[test]
fn chunk_copies_refuse_a_short_buffer() {
    let gpu = MockGpuBackend::new();
    let plan = ChunkPlan {
        entries: vec![(DevicePtr(4096), 0, 8)],
        bytes: 8,
    };
    let mut short = [0u8; 7];
    assert!(gather_chunk(&gpu, &plan, &mut short, 0).is_err());
    assert!(scatter_chunk(&gpu, &plan, &short, 0, false).is_err());
}

// ── swap image ──

/// KV regions of `table` followed by two SSM-like regions, and the legacy image bytes.
fn image_fixture(
    gpu: &MockGpuBackend,
    kv: &mut PagedKvCache,
    blocks_n: usize,
) -> (Vec<u32>, Vec<(DevicePtr, usize)>, Vec<u8>) {
    let table = alloc_table(kv, blocks_n);
    fill(gpu, kv, &table);
    let mut regions = flat(&kv.table_regions(&table).unwrap());
    let mut image = legacy_kv_bytes(gpu, kv, &table);
    for (i, n) in [(0u64, 3000usize), (1, 777)] {
        let p = gpu.alloc(n).unwrap();
        let bytes = pattern(90_000 + i, n);
        gpu.copy_h2d(&bytes, p).unwrap();
        regions.push((p, n));
        image.extend_from_slice(&bytes);
    }
    (table, regions, image)
}

#[test]
fn the_staged_image_equals_the_legacy_file_for_any_chunk_size() {
    for half in [7usize, 512, 4096, 1 << 20] {
        for (batched, events) in [(false, false), (true, false), (true, true)] {
            let gpu = MockGpuBackend::new();
            gpu.set_copy_batch_supported(batched);
            gpu.set_unique_events(events);
            gpu.hold_events(events);
            let mut kv = PagedKvCache::new(config(KvCacheDtype::Bf16, 3), 12, &gpu).unwrap();
            let (_t, regions, want) = image_fixture(&gpu, &mut kv, 5);
            let staging = SwapStaging::default();
            let mut file = Vec::new();
            let rep = write_image_with(&gpu, &regions, &staging, &mut file, 0, half).unwrap();
            assert_eq!(file, want, "half={half} batched={batched} events={events}");
            assert_eq!(rep.bytes, want.len());
            assert_eq!(rep.regions, regions.len());
            // Chunks are cut at the requested half; the buffer itself is page-rounded.
            assert_eq!(rep.chunks, want.len().div_ceil(half));
            assert_eq!(staging.allocated_bytes(), 2 * half.div_ceil(4096) * 4096);
            // Every recorded event was waited for before its bytes were written.
            assert_eq!(gpu.pending_event_count(), 0);
            staging.free(&gpu);
            assert_eq!(staging.allocated_bytes(), 0);
        }
    }
}

#[test]
fn restore_reads_a_legacy_file_and_a_new_file_alike() {
    for half in [4096usize, 1 << 20] {
        let gpu = MockGpuBackend::new();
        gpu.set_copy_batch_supported(true);
        gpu.set_unique_events(true);
        gpu.hold_events(true);
        let mut kv = PagedKvCache::new(config(KvCacheDtype::Fp8, 4), 24, &gpu).unwrap();
        let (_src, _regions, legacy_file) = image_fixture(&gpu, &mut kv, 6);
        // Restore into fresh blocks and fresh SSM buffers.
        let dst = alloc_table(&mut kv, 6);
        let mut regions = flat(&kv.table_regions(&dst).unwrap());
        let ssm: Vec<(DevicePtr, usize)> = [3000usize, 777]
            .iter()
            .map(|&n| (gpu.alloc(n).unwrap(), n))
            .collect();
        regions.extend(&ssm);
        let staging = SwapStaging::default();
        let rep = read_image_with(
            &gpu,
            &regions,
            &staging,
            &mut Cursor::new(&legacy_file),
            0,
            half,
        )
        .unwrap();
        assert_eq!(rep.bytes, legacy_file.len());
        let mut got = legacy_kv_bytes(&gpu, &kv, &dst);
        for &(p, n) in &ssm {
            let mut b = vec![0u8; n];
            gpu.copy_d2h(p, &mut b).unwrap();
            got.extend_from_slice(&b);
        }
        assert_eq!(got, legacy_file, "half={half}");
        // And the batched writer reproduces the same file from the restored state.
        let mut again = Vec::new();
        write_image_with(&gpu, &regions, &staging, &mut again, 0, half).unwrap();
        assert_eq!(again, legacy_file);
        staging.free(&gpu);
    }
}

#[test]
fn a_short_file_is_an_error_and_leaves_no_copy_in_flight() {
    let gpu = MockGpuBackend::new();
    gpu.set_unique_events(true);
    gpu.hold_events(true);
    let mut kv = PagedKvCache::new(config(KvCacheDtype::Fp8, 2), 8, &gpu).unwrap();
    let (_t, regions, mut file) = image_fixture(&gpu, &mut kv, 3);
    file.truncate(file.len() - 10);
    let staging = SwapStaging::default();
    let err = read_image_with(&gpu, &regions, &staging, &mut Cursor::new(&file), 0, 4096);
    assert!(err.is_err());
    // The staging is free for the next user (the lock was released after the sync).
    assert!(staging.acquire(&gpu, 4096).is_ok());
    staging.free(&gpu);
}

#[test]
fn an_empty_image_writes_nothing_and_allocates_nothing() {
    let gpu = MockGpuBackend::new();
    let staging = SwapStaging::default();
    let mut file = Vec::new();
    let rep = write_image_with(&gpu, &[], &staging, &mut file, 0, 4096).unwrap();
    assert!(file.is_empty());
    assert_eq!(rep.chunks, 0);
    assert_eq!(rep.staging, "none");
    assert_eq!(staging.allocated_bytes(), 0);
    let rep = read_image_with(&gpu, &[], &staging, &mut Cursor::new(&file), 0, 4096).unwrap();
    assert_eq!(rep.bytes, 0);
}

#[test]
fn staging_grows_to_the_largest_request_and_is_reused() {
    let gpu = MockGpuBackend::new();
    let s = SwapStaging::default();
    drop(s.acquire(&gpu, 4096).unwrap());
    assert_eq!(s.allocated_bytes(), 2 * 4096);
    drop(s.acquire(&gpu, 100).unwrap());
    assert_eq!(
        s.allocated_bytes(),
        2 * 4096,
        "a smaller request reuses the buffer"
    );
    drop(s.acquire(&gpu, 5000).unwrap());
    assert_eq!(
        s.allocated_bytes(),
        2 * 8192,
        "a larger one grows it (page multiple)"
    );
    assert!(s.acquire(&gpu, 0).is_err());
    s.free(&gpu);
    s.free(&gpu);
    assert_eq!(s.allocated_bytes(), 0);
}

#[test]
fn staging_half_is_clamped_to_one_pool_chunk() {
    const MIB: usize = 1 << 20;
    assert_eq!(staging_half_bytes(1, 64 * MIB), MIB);
    assert_eq!(staging_half_bytes(3 * MIB + 1, 64 * MIB), 4 * MIB);
    assert_eq!(staging_half_bytes(1 << 34, 64 * MIB), 32 * MIB);
    // A chunk smaller than two minimum halves still yields the minimum half.
    assert_eq!(staging_half_bytes(1 << 30, MIB), MIB);
}

#[test]
fn counters_and_report_lines_have_the_documented_shape() {
    let (before, _) = totals(SwapDir::Out);
    let gpu = MockGpuBackend::new();
    let mut kv = PagedKvCache::new(config(KvCacheDtype::Fp8, 1), 4, &gpu).unwrap();
    let (_t, regions, want) = image_fixture(&gpu, &mut kv, 2);
    let staging = SwapStaging::default();
    let mut file = Vec::new();
    let rep = write_image_with(&gpu, &regions, &staging, &mut file, 0, 4096).unwrap();
    staging.free(&gpu);
    let (after, secs) = totals(SwapDir::Out);
    // Other tests count concurrently: the counter moved by at least this image.
    assert!(after >= before + want.len() as u64);
    assert!(secs >= 0.0);

    let mut out = String::new();
    render_prometheus(&mut out);
    for needle in [
        "# TYPE atlas_kv_swap_bytes_total counter",
        "atlas_kv_swap_bytes_total{dir=\"d2h\"}",
        "atlas_kv_swap_bytes_total{dir=\"h2d\"}",
        "# TYPE atlas_kv_swap_seconds_total counter",
        "atlas_kv_swap_seconds_total{dir=\"h2d\"}",
    ] {
        assert!(out.contains(needle), "missing {needle:?} in\n{out}");
    }
    assert!(out.ends_with('\n'));

    let line = format_report(SwapDir::Out, "batched", 1000, &rep);
    assert!(line.starts_with("KV swap out (batched): "), "{line}");
    assert!(line.contains("GB/s"));
    assert!(!line.contains('\u{2014}'), "no em dash in log lines");
    let zero = SwapReport::default();
    assert_eq!(zero.gb_per_s(), 0.0);
    assert_eq!(SwapDir::In.label(), "h2d");
    assert_eq!(SwapDir::Out.label(), "d2h");
}
