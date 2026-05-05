// SPDX-License-Identifier: AGPL-3.0-only
//
// Shared test fixtures for the spark-storage integration tests. Only used
// from `tests/*.rs`; not part of the library surface.

#![allow(dead_code)]

use std::ffi::c_void;
use std::path::PathBuf;

use half::bf16;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use spark_storage::backend::PosixBackend;
use spark_storage::group::{GroupKey, GroupLayout, KvKind};
use spark_storage::layout::Layout;

pub const NUM_LAYERS: u32 = 1;
pub const NUM_SEQS: usize = 1;
pub const NUM_Q_HEADS: usize = 32;
pub const NUM_KV_HEADS: u16 = 8;
pub const HEAD_DIM: u32 = 128;
pub const BLOCK_SIZE: u32 = 16;
pub const NUM_BLOCKS: u32 = 8;
pub const FS_BLOCK: u64 = 4096;

pub fn random_bf16(n: usize, rng: &mut ChaCha8Rng) -> Vec<bf16> {
    let dist = StandardNormal;
    let inv = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    (0..n)
        .map(|_| {
            let v: f32 = dist.sample(rng);
            bf16::from_f32(v * inv)
        })
        .collect()
}

pub fn tempdir(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("atlas-storage-e2e-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

pub fn populate_disk(backend: &mut PosixBackend, k: &[bf16], v: &[bf16]) {
    use spark_storage::backend::StorageBackend;
    let head_dim = HEAD_DIM as usize;
    let bs = BLOCK_SIZE as usize;
    let nkv = NUM_KV_HEADS as usize;
    for blk in 0..NUM_BLOCKS {
        for kh in 0..NUM_KV_HEADS {
            let mut k_stripe = Vec::with_capacity(bs * head_dim);
            let mut v_stripe = Vec::with_capacity(bs * head_dim);
            for tok in 0..bs {
                let base = (blk as usize * bs * nkv + tok * nkv + kh as usize) * head_dim;
                k_stripe.extend_from_slice(&k[base..base + head_dim]);
                v_stripe.extend_from_slice(&v[base..base + head_dim]);
            }
            let group_bytes = backend.layout().group_bytes() as usize;
            let mut k_padded = vec![0_u8; group_bytes];
            let mut v_padded = vec![0_u8; group_bytes];
            let raw = bs * head_dim * 2;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    k_stripe.as_ptr() as *const u8,
                    k_padded.as_mut_ptr(),
                    raw,
                );
                std::ptr::copy_nonoverlapping(
                    v_stripe.as_ptr() as *const u8,
                    v_padded.as_mut_ptr(),
                    raw,
                );
            }
            backend
                .write_from_host(GroupKey::new(0, blk, kh, KvKind::K), &k_padded)
                .unwrap();
            backend
                .write_from_host(GroupKey::new(0, blk, kh, KvKind::V), &v_padded)
                .unwrap();
        }
    }
}

pub fn build_setup(seed: u64, dir: &PathBuf) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>, PosixBackend) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let q = random_bf16(NUM_SEQS * NUM_Q_HEADS * HEAD_DIM as usize, &mut rng);
    let total =
        NUM_BLOCKS as usize * BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let k = random_bf16(total, &mut rng);
    let v = random_bf16(total, &mut rng);
    let spec = GroupLayout::new(
        NUM_LAYERS,
        NUM_BLOCKS,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        HEAD_DIM,
        2,
        FS_BLOCK,
    );
    let layout = Layout::create(dir, spec).unwrap();
    let mut backend = PosixBackend::new(layout).unwrap();
    populate_disk(&mut backend, &k, &v);
    backend.drop_pagecache();
    (q, k, v, backend)
}

pub fn build_setup_iouring(
    seed: u64,
    dir: &PathBuf,
    qd: usize,
) -> (
    Vec<bf16>,
    Vec<bf16>,
    Vec<bf16>,
    spark_storage::backend::IoUringBackend,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let q = random_bf16(NUM_SEQS * NUM_Q_HEADS * HEAD_DIM as usize, &mut rng);
    let total =
        NUM_BLOCKS as usize * BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let k = random_bf16(total, &mut rng);
    let v = random_bf16(total, &mut rng);
    let spec = GroupLayout::new(
        NUM_LAYERS,
        NUM_BLOCKS,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        HEAD_DIM,
        2,
        FS_BLOCK,
    );
    let layout = Layout::create(dir, spec).unwrap();
    let mut backend = spark_storage::backend::IoUringBackend::new(layout, qd).unwrap();
    populate_disk_via(&mut backend, &k, &v);
    backend.drop_pagecache();
    (q, k, v, backend)
}

fn populate_disk_via<B: spark_storage::backend::StorageBackend + ?Sized>(
    backend: &mut B,
    k: &[bf16],
    v: &[bf16],
) {
    use spark_storage::group::KvKind;
    let head_dim = HEAD_DIM as usize;
    let bs = BLOCK_SIZE as usize;
    let nkv = NUM_KV_HEADS as usize;
    for blk in 0..NUM_BLOCKS {
        for kh in 0..NUM_KV_HEADS {
            let mut k_stripe = Vec::with_capacity(bs * head_dim);
            let mut v_stripe = Vec::with_capacity(bs * head_dim);
            for tok in 0..bs {
                let base = (blk as usize * bs * nkv + tok * nkv + kh as usize) * head_dim;
                k_stripe.extend_from_slice(&k[base..base + head_dim]);
                v_stripe.extend_from_slice(&v[base..base + head_dim]);
            }
            // O_DIRECT writes need page-aligned size buffers; pad to group_stride.
            // For our config the raw stripe size equals group_stride so no padding.
            let raw = bs * head_dim * 2;
            let mut k_padded = vec![0_u8; raw];
            let mut v_padded = vec![0_u8; raw];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    k_stripe.as_ptr() as *const u8,
                    k_padded.as_mut_ptr(),
                    raw,
                );
                std::ptr::copy_nonoverlapping(
                    v_stripe.as_ptr() as *const u8,
                    v_padded.as_mut_ptr(),
                    raw,
                );
            }
            backend
                .write_from_host(GroupKey::new(0, blk, kh, KvKind::K), &k_padded)
                .unwrap();
            backend
                .write_from_host(GroupKey::new(0, blk, kh, KvKind::V), &v_padded)
                .unwrap();
        }
    }
}

pub fn diff_stats(a: &[bf16], b: &[bf16]) -> (f32, f32) {
    let mut max_d = 0.0_f32;
    let mut sum_d = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x.to_f32() - y.to_f32()).abs();
        if d > max_d {
            max_d = d;
        }
        sum_d += d;
    }
    (max_d, sum_d / a.len() as f32)
}

#[allow(unused)]
pub fn _shut_up_unused_void(_: *const c_void) {}
