// SPDX-License-Identifier: AGPL-3.0-only
//
// Kernel-vs-reference parity test for the predictor. For each of the three
// kernels, generates random inputs, runs both the GPU kernel and the pure
// Rust reference, and asserts the outputs agree within BF16 tolerance.
//
// Tolerance rationale: BF16 has ~7 bits of mantissa. Reductions over
// `head_dim=128` accumulate error in the fma path; the kernel keeps a
// float accumulator before storing. We expect max abs diff <~ 2e-2 in
// projected outputs and <~ 5e-2 on dot products of those.

use std::ffi::c_void;

use half::bf16;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use spark_storage::cuda_min::{
    CudaCtx, DeviceBuffer, copy_d_to_h_async, copy_h_to_d_async, stream_sync,
};
use spark_storage::predictor::{Predictor, PredictorDims, read_k_lr_slot};
use spark_storage::predictor_ref::{predictor_score_ref, project_kv_block_ref, project_q_ref};
use spark_storage::projection::{PredictorShape, build_projection};

const NUM_LAYERS: usize = 4;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const R: usize = 32;
const BLOCK_SIZE: usize = 16;
const MAX_BLOCKS: usize = 64;

fn dims() -> PredictorDims {
    PredictorDims {
        num_layers: NUM_LAYERS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        r: R,
        block_size: BLOCK_SIZE,
        max_blocks: MAX_BLOCKS,
    }
}

fn random_bf16(n: usize, seed: u64) -> Vec<bf16> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dist = StandardNormal;
    let inv = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    (0..n)
        .map(|_| {
            let v: f32 = dist.sample(&mut rng);
            bf16::from_f32(v * inv)
        })
        .collect()
}

fn max_abs_bf16(a: &[bf16], b: &[bf16]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
        .fold(0.0_f32, f32::max)
}
fn max_abs_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
#[ignore = "requires GPU"]
fn q_lowrank_project_parity() {
    let ctx = CudaCtx::new(0).expect("cuda init");
    let pred = Predictor::new(&ctx, dims(), 0xCAFE_F00D).unwrap();
    let p_host = build_projection(PredictorShape::new(HEAD_DIM, R), 0xCAFE_F00D);
    let q_host = random_bf16(NUM_Q_HEADS * HEAD_DIM, 1);
    let q_dev = DeviceBuffer::new(q_host.len() * 2).unwrap();
    let q_proj_dev = DeviceBuffer::new(NUM_Q_HEADS * R * 2).unwrap();
    copy_h_to_d_async(
        q_dev.ptr,
        q_host.as_ptr() as *const c_void,
        q_host.len() * 2,
        ctx.stream,
    )
    .unwrap();
    pred.project_q(&ctx, q_dev.ptr, q_proj_dev.ptr).unwrap();
    let mut q_proj_gpu = vec![bf16::from_f32(0.0); NUM_Q_HEADS * R];
    copy_d_to_h_async(
        q_proj_gpu.as_mut_ptr() as *mut c_void,
        q_proj_dev.ptr,
        q_proj_gpu.len() * 2,
        ctx.stream,
    )
    .unwrap();
    stream_sync(ctx.stream).unwrap();
    let q_proj_ref = project_q_ref(&q_host, &p_host, NUM_Q_HEADS, HEAD_DIM, R);
    let diff = max_abs_bf16(&q_proj_gpu, &q_proj_ref);
    assert!(diff < 5e-2, "q_lowrank max abs diff = {diff}");
}

#[test]
#[ignore = "requires GPU"]
fn kv_lowrank_project_parity() {
    let ctx = CudaCtx::new(0).expect("cuda init");
    let pred = Predictor::new(&ctx, dims(), 0xCAFE_F00D).unwrap();
    let p_host = build_projection(PredictorShape::new(HEAD_DIM, R), 0xCAFE_F00D);
    let k_block = random_bf16(BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM, 7);
    let k_dev = DeviceBuffer::new(k_block.len() * 2).unwrap();
    copy_h_to_d_async(
        k_dev.ptr,
        k_block.as_ptr() as *const c_void,
        k_block.len() * 2,
        ctx.stream,
    )
    .unwrap();
    pred.project_kv_block(&ctx, /*layer=*/ 1, /*block_id=*/ 3, k_dev.ptr)
        .unwrap();
    let k_lr_gpu = read_k_lr_slot(&ctx, &pred, 1, 3).unwrap();
    let k_lr_ref = project_kv_block_ref(&k_block, &p_host, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, R);
    let diff = max_abs_bf16(&k_lr_gpu, &k_lr_ref);
    assert!(diff < 5e-2, "kv_lowrank max abs diff = {diff}");
}

#[test]
#[ignore = "requires GPU"]
fn predictor_score_parity() {
    let ctx = CudaCtx::new(0).expect("cuda init");
    let pred = Predictor::new(&ctx, dims(), 0xCAFE_F00D).unwrap();
    let n_active = 16;
    let q_proj = random_bf16(NUM_Q_HEADS * R, 11);
    let k_lr_seq = random_bf16(n_active * NUM_KV_HEADS * BLOCK_SIZE * R, 13);
    let q_proj_dev = DeviceBuffer::new(q_proj.len() * 2).unwrap();
    let k_lr_dev = DeviceBuffer::new(k_lr_seq.len() * 2).unwrap();
    let scores_dev = DeviceBuffer::new(n_active * 4).unwrap();
    copy_h_to_d_async(
        q_proj_dev.ptr,
        q_proj.as_ptr() as *const c_void,
        q_proj.len() * 2,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        k_lr_dev.ptr,
        k_lr_seq.as_ptr() as *const c_void,
        k_lr_seq.len() * 2,
        ctx.stream,
    )
    .unwrap();
    pred.score_blocks(&ctx, q_proj_dev.ptr, k_lr_dev.ptr, scores_dev.ptr, n_active)
        .unwrap();
    let mut scores_gpu = vec![0.0_f32; n_active];
    copy_d_to_h_async(
        scores_gpu.as_mut_ptr() as *mut c_void,
        scores_dev.ptr,
        n_active * 4,
        ctx.stream,
    )
    .unwrap();
    stream_sync(ctx.stream).unwrap();
    let scores_ref = predictor_score_ref(
        &q_proj,
        &k_lr_seq,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        R,
        n_active,
    );
    let diff = max_abs_f32(&scores_gpu, &scores_ref);
    let max_mag = scores_ref.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let rel = diff / max_mag.max(1e-6);
    assert!(
        rel < 5e-2,
        "score max abs diff = {diff}, rel = {rel}, scores: gpu={scores_gpu:?} ref={scores_ref:?}"
    );
}
