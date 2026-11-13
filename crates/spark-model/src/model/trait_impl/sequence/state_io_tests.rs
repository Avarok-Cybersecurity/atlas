// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU oracles for the sequence swap image on the mock backend: the per-region
//! writer (row `kv_batched_swap` off) and the batched writer (row on) produce the same file,
//! each reader reads the other's file, a save, free and restore reproduces every KV and SSM
//! byte, and a failed batched restore leaves no block allocated.

use std::io::Cursor;

use anyhow::anyhow;
use parking_lot::Mutex;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::block_io::swap::SwapStaging;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use super::*;

const BLOCKS: usize = 24;

fn config(dtype: KvCacheDtype) -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 64,
        num_layers: 3,
        dtype,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

fn pattern(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x2545_f491_4f6c_dd1d) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 16) as u8
        })
        .collect()
}

/// A cache with a filled block table and three GDN layers' worth of filled SSM buffers.
struct Fixture {
    gpu: MockGpuBackend,
    kv: Mutex<PagedKvCache>,
    table: Vec<u32>,
    ssm: Vec<SsmRegion>,
}

impl Fixture {
    fn new(dtype: KvCacheDtype, blocks: usize, batched_copies: bool) -> Self {
        let gpu = MockGpuBackend::new();
        gpu.set_copy_batch_supported(batched_copies);
        let mut kv = PagedKvCache::new(config(dtype), BLOCKS, &gpu).unwrap();
        let table: Vec<u32> = (0..blocks)
            .map(|_| kv.alloc_block().unwrap())
            .rev()
            .collect();
        for &b in &table {
            for l in 0..kv.num_layers() {
                let k = pattern(
                    u64::from(b) * 64 + l as u64 * 2,
                    kv.k_block_stride_bytes_for_layer(l),
                );
                let v = pattern(
                    u64::from(b) * 64 + l as u64 * 2 + 1,
                    kv.v_block_stride_bytes_for_layer(l),
                );
                kv.write_block(l, b, &k, &v, &gpu).unwrap();
            }
        }
        let ssm = (0..3u64).map(|i| Self::ssm_region(&gpu, i, true)).collect();
        Self {
            gpu,
            kv: Mutex::new(kv),
            table,
            ssm,
        }
    }

    /// One GDN layer's buffers, filled with a pattern (or zeroed).
    fn ssm_region(gpu: &MockGpuBackend, i: u64, filled: bool) -> SsmRegion {
        let (h_bytes, conv_bytes) = (8192 + 4 * i as usize, 1536);
        let h = gpu.alloc(h_bytes).unwrap();
        let conv = gpu.alloc(conv_bytes).unwrap();
        if filled {
            gpu.copy_h2d(&pattern(500 + i, h_bytes), h).unwrap();
            gpu.copy_h2d(&pattern(600 + i, conv_bytes), conv).unwrap();
        } else {
            gpu.memset(h, 0, h_bytes).unwrap();
            gpu.memset(conv, 0, conv_bytes).unwrap();
        }
        SsmRegion {
            h,
            h_bytes,
            conv,
            conv_bytes,
        }
    }

    fn fresh_ssm(&self) -> Vec<SsmRegion> {
        (0..3u64)
            .map(|i| Self::ssm_region(&self.gpu, i, false))
            .collect()
    }

    fn save(&self, batched: bool, staging: &SwapStaging) -> Vec<u8> {
        let mut file = Vec::new();
        let ssm = self.ssm.iter().copied().map(Ok);
        if batched {
            save_batched(&self.gpu, &self.kv, &self.table, ssm, staging, &mut file).unwrap();
        } else {
            save_legacy(&self.gpu, &self.kv, &self.table, ssm, &mut file).unwrap();
        }
        file
    }

    /// The image of `table` and `ssm` as the per-region reader sees it, read straight back.
    fn image_of(&self, table: &[u32], ssm: &[SsmRegion]) -> Vec<u8> {
        let kv = self.kv.lock();
        let mut out = Vec::new();
        for &b in table {
            for l in 0..kv.num_layers() {
                let (k, v) = kv.read_block(l, b, &self.gpu).unwrap();
                out.extend_from_slice(&k);
                out.extend_from_slice(&v);
            }
        }
        for r in ssm {
            for (p, n) in [(r.h, r.h_bytes), (r.conv, r.conv_bytes)] {
                let mut b = vec![0u8; n];
                self.gpu.copy_d2h(p, &mut b).unwrap();
                out.extend_from_slice(&b);
            }
        }
        out
    }

    fn free_blocks(&self) -> usize {
        self.kv.lock().num_free_blocks()
    }
}

#[test]
fn row_off_and_row_on_write_the_same_file() {
    for dtype in [
        KvCacheDtype::Bf16,
        KvCacheDtype::Fp8,
        KvCacheDtype::Bf16KTurbo3V,
    ] {
        for batched_copies in [false, true] {
            let f = Fixture::new(dtype, 7, batched_copies);
            let staging = SwapStaging::default();
            let off = f.save(false, &staging);
            let on = f.save(true, &staging);
            assert_eq!(on, off, "{dtype} batched_copies={batched_copies}");
            assert_eq!(
                on,
                f.image_of(&f.table, &f.ssm),
                "and both are the documented order"
            );
            staging.free(&f.gpu);
        }
    }
}

#[test]
fn either_reader_reads_either_file() {
    for dtype in [KvCacheDtype::Bf16, KvCacheDtype::Fp8] {
        for (write_batched, read_batched) in [(false, true), (true, false), (true, true)] {
            let f = Fixture::new(dtype, 5, true);
            let staging = SwapStaging::default();
            let file = f.save(write_batched, &staging);
            let ssm = f.fresh_ssm();
            let mut table = Vec::new();
            let mut reader = Cursor::new(&file);
            let it = ssm.iter().copied().map(Ok);
            if read_batched {
                restore_batched(&f.gpu, &f.kv, 5, &mut table, it, &staging, &mut reader).unwrap();
            } else {
                restore_legacy(&f.gpu, &f.kv, 5, &mut table, it, &mut reader).unwrap();
            }
            assert_eq!(table.len(), 5);
            assert_eq!(
                f.image_of(&table, &ssm),
                file,
                "{dtype} write_batched={write_batched} read_batched={read_batched}"
            );
            assert_eq!(
                reader.position() as usize,
                file.len(),
                "the whole file was consumed"
            );
            staging.free(&f.gpu);
        }
    }
}

#[test]
fn batched_restore_reads_asymmetric_dtypes_at_their_own_strides() {
    let f = Fixture::new(KvCacheDtype::Bf16KTurbo3V, 4, true);
    let staging = SwapStaging::default();
    let file = f.save(true, &staging);
    let ssm = f.fresh_ssm();
    let mut table = Vec::new();
    let it = ssm.iter().copied().map(Ok);
    restore_batched(
        &f.gpu,
        &f.kv,
        4,
        &mut table,
        it,
        &staging,
        &mut Cursor::new(&file),
    )
    .unwrap();
    assert_eq!(f.image_of(&table, &ssm), file);
    staging.free(&f.gpu);
}

#[test]
fn save_free_restore_reproduces_every_kv_and_ssm_byte() {
    let f = Fixture::new(KvCacheDtype::Bf16, 9, true);
    let staging = SwapStaging::default();
    let want = f.image_of(&f.table, &f.ssm);
    let file = f.save(true, &staging);
    // Swap out: the blocks go back to the pool and the SSM slot is cleared.
    let free_before = f.free_blocks();
    f.kv.lock().free_blocks(&f.table);
    assert_eq!(f.free_blocks(), free_before + 9);
    for r in &f.ssm {
        f.gpu.memset(r.h, 0, r.h_bytes).unwrap();
        f.gpu.memset(r.conv, 0, r.conv_bytes).unwrap();
    }
    // Swap in.
    let mut table = Vec::new();
    let it = f.ssm.iter().copied().map(Ok);
    let rep = restore_batched(
        &f.gpu,
        &f.kv,
        9,
        &mut table,
        it,
        &staging,
        &mut Cursor::new(&file),
    )
    .unwrap();
    assert_eq!(rep.bytes, want.len());
    assert_eq!(f.free_blocks(), free_before);
    assert_eq!(f.image_of(&table, &f.ssm), want);
    staging.free(&f.gpu);
}

#[test]
fn a_failed_batched_restore_frees_its_blocks_and_keeps_the_table() {
    let f = Fixture::new(KvCacheDtype::Fp8, 4, true);
    let staging = SwapStaging::default();
    let mut file = f.save(true, &staging);
    file.truncate(file.len() - 1);
    let free_before = f.free_blocks();
    let mut table = vec![77u32];
    let ssm = f.fresh_ssm();
    let it = ssm.iter().copied().map(Ok);
    let err = restore_batched(
        &f.gpu,
        &f.kv,
        4,
        &mut table,
        it,
        &staging,
        &mut Cursor::new(&file),
    );
    assert!(err.is_err());
    assert_eq!(
        f.free_blocks(),
        free_before,
        "every block it allocated went back"
    );
    assert_eq!(
        table,
        vec![77],
        "the caller's table is untouched on failure"
    );
    staging.free(&f.gpu);
}

#[test]
fn an_exhausted_pool_is_an_error_without_a_leak() {
    let f = Fixture::new(KvCacheDtype::Fp8, 4, true);
    let staging = SwapStaging::default();
    let file = f.save(true, &staging);
    let free_before = f.free_blocks();
    let mut table = Vec::new();
    let it = f.ssm.iter().copied().map(Ok);
    let too_many = free_before + 1;
    let err = restore_batched(
        &f.gpu,
        &f.kv,
        too_many,
        &mut table,
        it,
        &staging,
        &mut Cursor::new(&file),
    );
    assert!(err.is_err());
    assert_eq!(f.free_blocks(), free_before);
    assert!(table.is_empty());
}

#[test]
fn a_bad_layer_state_fails_the_batched_save_before_any_byte() {
    let f = Fixture::new(KvCacheDtype::Fp8, 2, true);
    let staging = SwapStaging::default();
    let mut file = Vec::new();
    let ssm = [
        Ok(f.ssm[0]),
        Err(anyhow!("Expected SsmLayerState at layer 3")),
    ]
    .into_iter();
    let err = save_batched(&f.gpu, &f.kv, &f.table, ssm, &staging, &mut file);
    assert!(err.is_err());
    assert!(file.is_empty());
    // The per-region writer meets it in phase 3, after the KV bytes: as before the sprint.
    let mut legacy = Vec::new();
    let ssm = [
        Ok(f.ssm[0]),
        Err(anyhow!("Expected SsmLayerState at layer 3")),
    ]
    .into_iter();
    assert!(save_legacy(&f.gpu, &f.kv, &f.table, ssm, &mut legacy).is_err());
    assert!(!legacy.is_empty());
}

#[test]
fn an_empty_sequence_is_an_empty_image() {
    let gpu = MockGpuBackend::new();
    let kv = Mutex::new(PagedKvCache::new(config(KvCacheDtype::Fp8), 4, &gpu).unwrap());
    let staging = SwapStaging::default();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    save_legacy(&gpu, &kv, &[], std::iter::empty(), &mut a).unwrap();
    save_batched(&gpu, &kv, &[], std::iter::empty(), &staging, &mut b).unwrap();
    assert!(a.is_empty() && b.is_empty());
    assert_eq!(staging.allocated_bytes(), 0);
    let mut table = Vec::new();
    restore_batched(
        &gpu,
        &kv,
        0,
        &mut table,
        std::iter::empty(),
        &staging,
        &mut Cursor::new(&b),
    )
    .unwrap();
    assert!(table.is_empty());
}
