// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-5 integration gate: the `HighSpeedSwap` orchestrator end-to-end.
//
// Builds a sequence with `seq_blocks > scratch_capacity` so the tile loop
// is forced to fetch from disk and evict prior-tile blocks. Output must
// match the in-HBM single-tile reference attention.

use std::ffi::c_void;
use std::path::PathBuf;

use half::bf16;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use spark_storage::cuda_min::{
    CudaCtx, DeviceBuffer, copy_d_to_h_async, copy_h_to_d_async, stream_sync,
};
use spark_storage::tiled_attention::{TiledAttention, TiledAttentionDims};
use spark_storage::{HighSpeedSwap, HighSpeedSwapConfig, ModelDims};

const NUM_LAYERS: u32 = 1;
const NUM_Q_HEADS: u16 = 32;
const NUM_KV_HEADS: u16 = 8;
const HEAD_DIM: u16 = 128;
const BLOCK_SIZE: u16 = 16;
const SEQ_BLOCKS: u32 = 32; // 32 blocks = 512 tokens
const SCRATCH_BLOCKS: u32 = 8; // 4× more blocks than scratch — forces eviction

fn random_bf16(n: usize, rng: &mut ChaCha8Rng) -> Vec<bf16> {
    let dist = StandardNormal;
    let inv = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    (0..n)
        .map(|_| {
            let v: f32 = dist.sample(rng);
            bf16::from_f32(v * inv)
        })
        .collect()
}

fn tempdir(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("atlas-hss-e2e-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn run_in_hbm_reference(ctx: &CudaCtx, q: &[bf16], k: &[bf16], v: &[bf16]) -> Vec<bf16> {
    let dims = TiledAttentionDims {
        max_seqs: 1,
        num_q_heads: NUM_Q_HEADS as usize,
        num_kv_heads: NUM_KV_HEADS as usize,
        head_dim: HEAD_DIM as usize,
        block_size: BLOCK_SIZE as usize,
        tile_capacity: SEQ_BLOCKS as usize,
    };
    let attn = TiledAttention::new(dims).unwrap();
    let q_dev = DeviceBuffer::new(q.len() * 2).unwrap();
    let k_dev = DeviceBuffer::new(k.len() * 2).unwrap();
    let v_dev = DeviceBuffer::new(v.len() * 2).unwrap();
    let bt: Vec<i32> = (0..SEQ_BLOCKS as i32).collect();
    let bt_dev = DeviceBuffer::new(SEQ_BLOCKS as usize * 4).unwrap();
    let counts_dev = DeviceBuffer::new(4).unwrap();
    let counts = [SEQ_BLOCKS as i32];
    let out_dev = DeviceBuffer::new(NUM_Q_HEADS as usize * HEAD_DIM as usize * 2).unwrap();
    copy_h_to_d_async(
        q_dev.ptr,
        q.as_ptr() as *const c_void,
        q.len() * 2,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        k_dev.ptr,
        k.as_ptr() as *const c_void,
        k.len() * 2,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        v_dev.ptr,
        v.as_ptr() as *const c_void,
        v.len() * 2,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        bt_dev.ptr,
        bt.as_ptr() as *const c_void,
        SEQ_BLOCKS as usize * 4,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        counts_dev.ptr,
        counts.as_ptr() as *const c_void,
        4,
        ctx.stream,
    )
    .unwrap();
    attn.begin_step(ctx, 1).unwrap();
    let (s_blk, s_tok, s_kvh) = attn.paged_strides();
    attn.step_tile(
        ctx,
        q_dev.ptr,
        k_dev.ptr,
        v_dev.ptr,
        bt_dev.ptr,
        counts_dev.ptr,
        1,
        s_blk,
        s_tok,
        s_kvh,
        BLOCK_SIZE as i32,
    )
    .unwrap();
    attn.finalize(ctx, out_dev.ptr, 1).unwrap();
    let mut out = vec![bf16::from_f32(0.0); NUM_Q_HEADS as usize * HEAD_DIM as usize];
    copy_d_to_h_async(
        out.as_mut_ptr() as *mut c_void,
        out_dev.ptr,
        out.len() * 2,
        ctx.stream,
    )
    .unwrap();
    stream_sync(ctx.stream).unwrap();
    out
}

#[test]
#[ignore = "requires GPU"]
fn orchestrator_multi_tile_with_eviction() {
    let dir = tempdir("multi-tile");
    let ctx = CudaCtx::new(0).expect("cuda init");
    let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
    let q = random_bf16(NUM_Q_HEADS as usize * HEAD_DIM as usize, &mut rng);
    let total =
        SEQ_BLOCKS as usize * BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let k = random_bf16(total, &mut rng);
    let v = random_bf16(total, &mut rng);

    // Reference: single in-HBM attention over the full sequence.
    let reference = run_in_hbm_reference(&ctx, &q, &k, &v);

    // Spin up the orchestrator with scratch sized to 1/4 of the sequence so
    // every step has to evict ~3 tile-fulls of cold data and stream them in.
    let cfg = HighSpeedSwapConfig {
        dir: dir.clone(),
        bytes: 1 << 30,
        resident_blocks: SCRATCH_BLOCKS,
        rank: 32,
        qd: 8,
        graph: false,
        projection_seed: 0xCAFE_F00D,
    };
    let model = ModelDims {
        num_layers: NUM_LAYERS,
        max_blocks_per_layer: SEQ_BLOCKS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        block_size: BLOCK_SIZE,
    };
    let mut hss = HighSpeedSwap::new(&ctx, cfg, model).unwrap();

    // Offload every block via the public API.
    let block_floats = BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let block_bytes = block_floats * 2;
    let k_block_dev = DeviceBuffer::new(block_bytes).unwrap();
    for blk in 0..SEQ_BLOCKS {
        let off = blk as usize * block_floats;
        copy_h_to_d_async(
            k_block_dev.ptr,
            k[off..off + block_floats].as_ptr() as *const c_void,
            block_bytes,
            ctx.stream,
        )
        .unwrap();
        stream_sync(ctx.stream).unwrap();
        hss.offload_block(
            &ctx,
            0,
            blk,
            k_block_dev.ptr,
            &k[off..off + block_floats],
            &v[off..off + block_floats],
        )
        .unwrap();
    }

    // Run streaming attention via the orchestrator.
    let q_dev = DeviceBuffer::new(q.len() * 2).unwrap();
    let out_dev = DeviceBuffer::new(NUM_Q_HEADS as usize * HEAD_DIM as usize * 2).unwrap();
    copy_h_to_d_async(
        q_dev.ptr,
        q.as_ptr() as *const c_void,
        q.len() * 2,
        ctx.stream,
    )
    .unwrap();
    let seq_blocks: Vec<u32> = (0..SEQ_BLOCKS).collect();
    hss.attend_layer(&ctx, 0, &seq_blocks, q_dev.ptr, out_dev.ptr)
        .unwrap();

    let mut out = vec![bf16::from_f32(0.0); NUM_Q_HEADS as usize * HEAD_DIM as usize];
    copy_d_to_h_async(
        out.as_mut_ptr() as *mut c_void,
        out_dev.ptr,
        out.len() * 2,
        ctx.stream,
    )
    .unwrap();
    stream_sync(ctx.stream).unwrap();

    let mut max_d = 0.0_f32;
    for (a, b) in reference.iter().zip(&out) {
        let d = (a.to_f32() - b.to_f32()).abs();
        if d > max_d {
            max_d = d;
        }
    }
    eprintln!(
        "seq_blocks={SEQ_BLOCKS} scratch_blocks={SCRATCH_BLOCKS} \
         tiles={} max abs diff = {max_d:.3e}",
        SEQ_BLOCKS.div_ceil(SCRATCH_BLOCKS)
    );
    assert!(
        max_d < 1e-2,
        "orchestrator output diverged from reference: {max_d}"
    );
    std::fs::remove_dir_all(&dir).ok();
}
