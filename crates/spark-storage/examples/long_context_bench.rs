// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-5 long-context throughput bench for the high-speed-swap orchestrator.
//
// Decode-style sustained-attention workload at varying context lengths:
//   1. Build a synthetic Q/K/V dataset matching one model layer's shape.
//   2. Offload every block to NVMe via `HighSpeedSwap::offload_block`.
//   3. Run N decode "steps", each calling `attend_layer` on the full
//      sequence (scratch can never hold the whole thing → tile loop +
//      eviction every step).
//   4. Compare to an in-HBM single-tile reference time (only sized to fit).
//
// Reports sustained tok/s + bandwidth, gives the user the data point the
// plan's 2-4× claim ultimately relies on.
//
// Run:
//   cargo run --release -p spark-storage --example long-context-bench -- \
//       --dir /tmp/atlas-hsw-bench --context-blocks 256 --scratch-blocks 32 \
//       --steps 64

use std::ffi::c_void;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
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

const NUM_Q_HEADS: u16 = 32;
const NUM_KV_HEADS: u16 = 8;
const HEAD_DIM: u16 = 128;
const BLOCK_SIZE: u16 = 16;

struct Args {
    dir: PathBuf,
    context_blocks: u32,
    scratch_blocks: u32,
    steps: usize,
}

fn parse_args() -> Args {
    let mut dir: Option<PathBuf> = None;
    let mut context_blocks: u32 = 64;
    let mut scratch_blocks: u32 = 16;
    let mut steps: usize = 32;
    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        match arg.as_str() {
            "--dir" => dir = Some(PathBuf::from(a.next().unwrap())),
            "--context-blocks" => context_blocks = a.next().unwrap().parse().unwrap(),
            "--scratch-blocks" => scratch_blocks = a.next().unwrap().parse().unwrap(),
            "--steps" => steps = a.next().unwrap().parse().unwrap(),
            "--help" | "-h" => {
                eprintln!(
                    "usage: long-context-bench --dir <p> [--context-blocks N] [--scratch-blocks N] [--steps N]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    Args {
        dir: dir.expect("--dir required"),
        context_blocks,
        scratch_blocks,
        steps,
    }
}

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

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = parse_args();
    let _ = std::fs::remove_dir_all(&args.dir);
    std::fs::create_dir_all(&args.dir)?;

    let ctx = CudaCtx::new(0)?;
    let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
    let q = random_bf16(NUM_Q_HEADS as usize * HEAD_DIM as usize, &mut rng);
    let total = args.context_blocks as usize
        * BLOCK_SIZE as usize
        * NUM_KV_HEADS as usize
        * HEAD_DIM as usize;
    let k = random_bf16(total, &mut rng);
    let v = random_bf16(total, &mut rng);

    let context_tokens = args.context_blocks as usize * BLOCK_SIZE as usize;
    eprintln!(
        "context: {} blocks × {} tokens = {} tokens; scratch = {} blocks ({}× over-subscription); steps = {}",
        args.context_blocks,
        BLOCK_SIZE,
        context_tokens,
        args.scratch_blocks,
        args.context_blocks / args.scratch_blocks.max(1),
        args.steps,
    );

    let cfg = HighSpeedSwapConfig {
        dir: args.dir.clone(),
        bytes: 64 * (1 << 30),
        resident_blocks: args.scratch_blocks,
        rank: 32,
        qd: 8,
        graph: false,
        projection_seed: 0xCAFE_F00D,
    };
    let model = ModelDims {
        num_layers: 1,
        max_blocks_per_layer: args.context_blocks,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        block_size: BLOCK_SIZE,
        model_fp: None,
    };
    let mut hss = HighSpeedSwap::new(&ctx, cfg, model)?;

    // Offload every block to disk.
    let block_floats = BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let block_bytes = block_floats * 2;
    let k_block_dev = DeviceBuffer::new(block_bytes)?;
    let t_offload = Instant::now();
    for blk in 0..args.context_blocks {
        let off = blk as usize * block_floats;
        copy_h_to_d_async(
            k_block_dev.ptr,
            k[off..off + block_floats].as_ptr() as *const c_void,
            block_bytes,
            ctx.stream,
        )?;
        stream_sync(ctx.stream)?;
        hss.offload_block(
            &ctx,
            0,
            blk,
            k_block_dev.ptr,
            &k[off..off + block_floats],
            &v[off..off + block_floats],
        )?;
    }
    let offload_dt = t_offload.elapsed().as_secs_f64();
    let offload_mib = (args.context_blocks as f64) * (block_bytes as f64) * 2.0 / (1024.0 * 1024.0);
    eprintln!(
        "offload: {} blocks ({:.1} MiB K+V) in {:.2}s = {:.1} MiB/s",
        args.context_blocks,
        offload_mib,
        offload_dt,
        offload_mib / offload_dt
    );

    let q_dev = DeviceBuffer::new(q.len() * 2)?;
    let out_dev = DeviceBuffer::new(NUM_Q_HEADS as usize * HEAD_DIM as usize * 2)?;
    copy_h_to_d_async(
        q_dev.ptr,
        q.as_ptr() as *const c_void,
        q.len() * 2,
        ctx.stream,
    )?;
    stream_sync(ctx.stream)?;
    let seq: Vec<u32> = (0..args.context_blocks).collect();

    // Warmup.
    for _ in 0..2 {
        hss.attend_layer(&ctx, 0, &seq, q_dev.ptr, out_dev.ptr)?;
    }
    stream_sync(ctx.stream)?;

    // Streaming bench.
    let t_stream = Instant::now();
    for _ in 0..args.steps {
        hss.attend_layer(&ctx, 0, &seq, q_dev.ptr, out_dev.ptr)?;
    }
    stream_sync(ctx.stream)?;
    let stream_dt = t_stream.elapsed().as_secs_f64();
    let per_step_us = stream_dt * 1e6 / args.steps as f64;
    let bytes_per_step = (args.context_blocks as f64) * (block_bytes as f64) * 2.0;
    let stream_gbps = bytes_per_step * args.steps as f64 / stream_dt / (1024.0 * 1024.0 * 1024.0);

    // In-HBM reference (only if sequence fits with attention dims).
    let in_hbm_us = run_in_hbm(&ctx, &q, &k, &v, args.context_blocks, args.steps).ok();

    eprintln!();
    eprintln!("== Streaming attention (HighSpeedSwap, scratch < context):");
    eprintln!("   per-step latency : {per_step_us:.0} µs");
    eprintln!("   sustained read   : {stream_gbps:.2} GiB/s");
    if let Some(hbm_us) = in_hbm_us {
        let ratio = per_step_us / hbm_us;
        eprintln!();
        eprintln!("== In-HBM reference (full context resident):");
        eprintln!("   per-step latency : {hbm_us:.0} µs");
        eprintln!(
            "   streaming overhead: {ratio:.2}× ({:.0} µs of NVMe streaming + tile loop)",
            per_step_us - hbm_us
        );
    }
    let _ = out_dev;
    Ok(())
}

fn run_in_hbm(
    ctx: &CudaCtx,
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    blocks: u32,
    steps: usize,
) -> Result<f64> {
    let dims = TiledAttentionDims {
        max_seqs: 1,
        num_q_heads: NUM_Q_HEADS as usize,
        num_kv_heads: NUM_KV_HEADS as usize,
        head_dim: HEAD_DIM as usize,
        block_size: BLOCK_SIZE as usize,
        tile_capacity: blocks as usize,
    };
    let attn = TiledAttention::new(dims)?;
    let q_dev = DeviceBuffer::new(q.len() * 2)?;
    let k_dev = DeviceBuffer::new(k.len() * 2)?;
    let v_dev = DeviceBuffer::new(v.len() * 2)?;
    let bt: Vec<i32> = (0..blocks as i32).collect();
    let bt_dev = DeviceBuffer::new(bt.len() * 4)?;
    let counts_dev = DeviceBuffer::new(4)?;
    let counts = [blocks as i32];
    let out_dev = DeviceBuffer::new(NUM_Q_HEADS as usize * HEAD_DIM as usize * 2)?;
    copy_h_to_d_async(
        q_dev.ptr,
        q.as_ptr() as *const c_void,
        q.len() * 2,
        ctx.stream,
    )?;
    copy_h_to_d_async(
        k_dev.ptr,
        k.as_ptr() as *const c_void,
        k.len() * 2,
        ctx.stream,
    )?;
    copy_h_to_d_async(
        v_dev.ptr,
        v.as_ptr() as *const c_void,
        v.len() * 2,
        ctx.stream,
    )?;
    copy_h_to_d_async(
        bt_dev.ptr,
        bt.as_ptr() as *const c_void,
        bt.len() * 4,
        ctx.stream,
    )?;
    copy_h_to_d_async(
        counts_dev.ptr,
        counts.as_ptr() as *const c_void,
        4,
        ctx.stream,
    )?;
    let (s_blk, s_tok, s_kvh) = attn.paged_strides();
    // Warmup.
    for _ in 0..2 {
        attn.begin_step(ctx, 1)?;
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
        )?;
        attn.finalize(ctx, out_dev.ptr, 1)?;
    }
    stream_sync(ctx.stream)?;
    let t = Instant::now();
    for _ in 0..steps {
        attn.begin_step(ctx, 1)?;
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
        )?;
        attn.finalize(ctx, out_dev.ptr, 1)?;
    }
    stream_sync(ctx.stream)?;
    let dt = t.elapsed().as_secs_f64();
    let _ = (q_dev, k_dev, v_dev, bt_dev, counts_dev, out_dev);
    let mut _swallow = vec![0_u8; 1];
    copy_d_to_h_async(
        _swallow.as_mut_ptr() as *mut c_void,
        ctx.stream,
        0,
        ctx.stream,
    )
    .ok();
    Ok(dt * 1e6 / steps as f64)
}
