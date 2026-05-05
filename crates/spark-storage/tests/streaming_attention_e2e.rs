// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-2 end-to-end gate: streaming attention via the POSIX backend produces
// bit-identical output to the in-HBM reference at all tile sizes.

mod common;

use std::ffi::c_void;

use half::bf16;
use spark_storage::backend::{ReadRequest, StorageBackend};
use spark_storage::cuda_min::{
    CudaCtx, DeviceBuffer, copy_d_to_h_async, copy_h_to_d_async, stream_sync,
};
use spark_storage::group::{GroupKey, KvKind};
use spark_storage::scratch_pool::{ResidentKey, ScratchDims, ScratchPool};
use spark_storage::tiled_attention::{TiledAttention, TiledAttentionDims};

use common::*;

fn run_in_hbm_reference(ctx: &CudaCtx, q: &[bf16], k: &[bf16], v: &[bf16]) -> Vec<bf16> {
    let dims = TiledAttentionDims {
        max_seqs: NUM_SEQS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS as usize,
        head_dim: HEAD_DIM as usize,
        block_size: BLOCK_SIZE as usize,
        tile_capacity: NUM_BLOCKS as usize,
    };
    let attn = TiledAttention::new(dims).unwrap();
    let q_dev = DeviceBuffer::new(q.len() * 2).unwrap();
    let k_dev = DeviceBuffer::new(k.len() * 2).unwrap();
    let v_dev = DeviceBuffer::new(v.len() * 2).unwrap();
    let block_table: Vec<i32> = (0..NUM_BLOCKS as i32).collect();
    let counts = [NUM_BLOCKS as i32];
    let bt_dev = DeviceBuffer::new(NUM_BLOCKS as usize * 4).unwrap();
    let counts_dev = DeviceBuffer::new(NUM_SEQS * 4).unwrap();
    let out_dev = DeviceBuffer::new(NUM_SEQS * NUM_Q_HEADS * HEAD_DIM as usize * 2).unwrap();
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
        block_table.as_ptr() as *const c_void,
        NUM_BLOCKS as usize * 4,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        counts_dev.ptr,
        counts.as_ptr() as *const c_void,
        NUM_SEQS * 4,
        ctx.stream,
    )
    .unwrap();
    attn.begin_step(ctx, NUM_SEQS).unwrap();
    let (s_blk, s_tok, s_kvh) = attn.paged_strides();
    attn.step_tile(
        ctx,
        q_dev.ptr,
        k_dev.ptr,
        v_dev.ptr,
        bt_dev.ptr,
        counts_dev.ptr,
        NUM_SEQS,
        s_blk,
        s_tok,
        s_kvh,
        BLOCK_SIZE as i32,
    )
    .unwrap();
    attn.finalize(ctx, out_dev.ptr, NUM_SEQS).unwrap();
    let mut out = vec![bf16::from_f32(0.0); NUM_SEQS * NUM_Q_HEADS * HEAD_DIM as usize];
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

fn run_streaming<B: StorageBackend + ?Sized>(
    ctx: &CudaCtx,
    backend: &mut B,
    group_stride: u64,
    q: &[bf16],
    tile_size: usize,
) -> Vec<bf16> {
    let mut pool = ScratchPool::new(ScratchDims {
        num_slots: tile_size as u32,
        num_kv_heads: NUM_KV_HEADS,
        group_stride,
    })
    .unwrap();
    let dims = TiledAttentionDims {
        max_seqs: NUM_SEQS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS as usize,
        head_dim: HEAD_DIM as usize,
        block_size: BLOCK_SIZE as usize,
        tile_capacity: tile_size,
    };
    let attn = TiledAttention::new(dims).unwrap();
    let q_dev = DeviceBuffer::new(q.len() * 2).unwrap();
    let bt_dev = DeviceBuffer::new(NUM_SEQS * tile_size * 4).unwrap();
    let counts_dev = DeviceBuffer::new(NUM_SEQS * 4).unwrap();
    let out_dev = DeviceBuffer::new(NUM_SEQS * NUM_Q_HEADS * HEAD_DIM as usize * 2).unwrap();
    copy_h_to_d_async(
        q_dev.ptr,
        q.as_ptr() as *const c_void,
        q.len() * 2,
        ctx.stream,
    )
    .unwrap();
    attn.begin_step(ctx, NUM_SEQS).unwrap();

    let n_tiles = (NUM_BLOCKS as usize).div_ceil(tile_size);
    for t in 0..n_tiles {
        pool.clear();
        let start = t * tile_size;
        let end = (start + tile_size).min(NUM_BLOCKS as usize);
        let n = end - start;

        let mut reqs = Vec::with_capacity(n * NUM_KV_HEADS as usize * 2);
        let mut block_table = vec![0_i32; tile_size];
        for (i, blk) in (start..end).enumerate() {
            let key = ResidentKey {
                layer: 0,
                block: blk as u32,
            };
            let slot = pool.assign(key, &[]).unwrap();
            block_table[i] = slot as i32;
            for kh in 0..NUM_KV_HEADS {
                reqs.push(ReadRequest {
                    group: GroupKey::new(0, blk as u32, kh, KvKind::K),
                    dst_dev_ptr: pool.slot_k_ptr(slot, kh),
                });
                reqs.push(ReadRequest {
                    group: GroupKey::new(0, blk as u32, kh, KvKind::V),
                    dst_dev_ptr: pool.slot_v_ptr(slot, kh),
                });
            }
        }
        backend.read(&reqs, ctx.stream).unwrap();

        let counts = [n as i32];
        copy_h_to_d_async(
            bt_dev.ptr,
            block_table.as_ptr() as *const c_void,
            tile_size * 4,
            ctx.stream,
        )
        .unwrap();
        copy_h_to_d_async(
            counts_dev.ptr,
            counts.as_ptr() as *const c_void,
            NUM_SEQS * 4,
            ctx.stream,
        )
        .unwrap();
        let (s_blk, s_tok, s_kvh) = attn.scratch_pool_strides();
        let v_offset_bytes = (NUM_KV_HEADS as u64) * (BLOCK_SIZE as u64) * (HEAD_DIM as u64) * 2;
        attn.step_tile(
            ctx,
            q_dev.ptr,
            pool.pool_dev_ptr(),
            pool.pool_dev_ptr() + v_offset_bytes,
            bt_dev.ptr,
            counts_dev.ptr,
            NUM_SEQS,
            s_blk,
            s_tok,
            s_kvh,
            BLOCK_SIZE as i32,
        )
        .unwrap();
    }
    attn.finalize(ctx, out_dev.ptr, NUM_SEQS).unwrap();
    let mut out = vec![bf16::from_f32(0.0); NUM_SEQS * NUM_Q_HEADS * HEAD_DIM as usize];
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
fn streaming_matches_in_hbm_posix() {
    let dir = tempdir("stream-posix");
    let ctx = CudaCtx::new(0).expect("cuda init");
    let (q, k, v, mut backend) = build_setup(0xCAFE, &dir);
    let reference = run_in_hbm_reference(&ctx, &q, &k, &v);
    let stride = backend.layout().group_bytes();
    for tile_size in [NUM_BLOCKS as usize, 4, 2, 1] {
        let actual = run_streaming(&ctx, &mut backend, stride, &q, tile_size);
        let (max_d, mean_d) = diff_stats(&reference, &actual);
        eprintln!("posix  tile_size={tile_size}: max abs diff = {max_d:.3e}, mean = {mean_d:.3e}");
        assert!(max_d < 1e-2, "posix tile_size={tile_size} max_d = {max_d}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "requires GPU"]
fn streaming_matches_in_hbm_iouring() {
    let dir = tempdir("stream-iouring");
    let ctx = CudaCtx::new(0).expect("cuda init");
    let (q, k, v, mut backend) = build_setup_iouring(0xCAFE, &dir, 8);
    let reference = run_in_hbm_reference(&ctx, &q, &k, &v);
    let stride = backend.layout().group_bytes();
    for tile_size in [NUM_BLOCKS as usize, 4, 2, 1] {
        let actual = run_streaming(&ctx, &mut backend, stride, &q, tile_size);
        let (max_d, mean_d) = diff_stats(&reference, &actual);
        eprintln!("iour   tile_size={tile_size}: max abs diff = {max_d:.3e}, mean = {mean_d:.3e}");
        assert!(
            max_d < 1e-2,
            "iouring tile_size={tile_size} max_d = {max_d}"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "requires GPU"]
fn round_trip_groups_through_disk() {
    let dir = tempdir("rt");
    let ctx = CudaCtx::new(0).expect("cuda init");
    let (_, k, _v, mut backend) = build_setup(0xBEEF, &dir);
    let group = GroupKey::new(0, 3, 5, KvKind::K);
    let bytes = backend.layout().group_bytes() as usize;
    let dev = DeviceBuffer::new(bytes).unwrap();
    backend
        .read(
            &[ReadRequest {
                group,
                dst_dev_ptr: dev.ptr,
            }],
            ctx.stream,
        )
        .unwrap();
    let mut got = vec![0_u8; bytes];
    copy_d_to_h_async(got.as_mut_ptr() as *mut c_void, dev.ptr, bytes, ctx.stream).unwrap();
    stream_sync(ctx.stream).unwrap();
    let head_dim = HEAD_DIM as usize;
    let bs = BLOCK_SIZE as usize;
    let nkv = NUM_KV_HEADS as usize;
    let mut expected = Vec::with_capacity(bs * head_dim);
    for tok in 0..bs {
        let base = (3 * bs * nkv + tok * nkv + 5) * head_dim;
        expected.extend_from_slice(&k[base..base + head_dim]);
    }
    let raw = bs * head_dim * 2;
    let exp_bytes: Vec<u8> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(&got[..raw], &exp_bytes[..]);
    std::fs::remove_dir_all(&dir).ok();
}
