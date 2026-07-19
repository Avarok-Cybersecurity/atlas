// SPDX-License-Identifier: AGPL-3.0-only

//! GDN prefill ops — extracted from `ssm_gdn_a.rs` during the ≤500-line split.
//! All public items remain available at `crate::layers::ops::*` via the
//! re-export in `ops.rs`.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Gated delta rule prefill (multi-token, sequential SSM update within kernel).
///
/// Processes `seq_len` tokens sequentially per (batch, head) pair.
/// Supports strided access: Q/K/V/gate/beta may have different strides
/// between tokens (e.g., from conv1d output with interleaved Q|K|V layout).
///
/// Kernel: `gated_delta_rule_prefill(h_state, query, key, value,
///          gate, beta, output, batch_size, seq_len, num_k_heads,
///          num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride)`
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(4 * k_dim * 4) // double-buffered k[128]+q[128] × 2 buffers × 4 bytes
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// Split-v_dim prefill: 2 CTAs per v-head, 64 threads each.
///
/// Kernel: `gated_delta_rule_prefill_split(h_state, query, key, value,
///          gate, beta, output, batch_size, seq_len, num_k_heads,
///          num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride)`
/// Grid: (num_v_heads * 2, batch, 1)  Block: (64, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_split(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads * 2, batch_size, 1])
        .block([64, 1, 1])
        .shared_mem(4 * k_dim * 4) // double-buffered k[K_DIM]+q[K_DIM] × 2 buffers × 4 bytes
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// 4-way split prefill: 4 CTAs per v-head, 32 threads each (128 total CTAs).
///
/// Kernel: `gated_delta_rule_prefill_split4(h_state, query, key, value,
///          gate, beta, output, batch_size, seq_len, num_k_heads,
///          num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride)`
/// Grid: (num_v_heads * 4, batch, 1)  Block: (32, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_split4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads * 4, batch_size, 1])
        .block([32, 1, 1])
        .shared_mem(4 * k_dim * 4) // double-buffered k[K_DIM]+q[K_DIM] × 2 buffers × 4 bytes
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// Persistent GDN prefill — h_state stays in shared memory for entire sequence.
///
/// Same parameters as gdn_prefill_split4 but uses persistent CTAs with
/// 128 threads and 67 KB shared memory. Each CTA processes ALL tokens for
/// one v_head, keeping h_state in shared memory (never written to global
/// until the end). Targets L2 bandwidth (~3 TB/s) instead of LPDDR5X (273 GB/s).
///
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
/// Shared: k_dim*v_dim*4 + 4*k_dim*4 bytes
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_persistent(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    let smem = k_dim * v_dim * 4 + 4 * k_dim * 4; // h_state + double-buffered k/q
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// Persistent GDN prefill with explicit shared memory size.
/// Used for WY4-persistent variant which needs more shared memory.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_persistent_smem(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    smem: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// FLA multi-kernel chunked GDN prefill (`ATLAS_GDN_FLA=1`).
///
/// Three sequential launches on `stream` (CPU-serialized → no GPU sync needed):
///   1. recompute_wu  (grid [num_chunks, nv, batch], 128 thr): solve (I+L)U=βV,
///      (I+L)W=β·exp(gc)·K → W_out, U_out (bf16), gc_out (f32).
///   2. chunk_delta_h_ksplit (grid [nv, batch], 256 thr): serial f32 state spine,
///      2 threads/v-column for occupancy → S_out (per-chunk entry states bf16),
///      uc_out (bf16); updates h_state in-place.
///   3. chunk_fwd_o   (grid [num_chunks, nv, batch], 128 thr): O = Q̃·S_c +
///      tril(decay·Q̃·Kᵀ)·uc → output (bf16, same layout as wy4).
///
/// W_out/U_out/S_out/uc_out are the caller's pre-sized scratch (BufferArena
/// `gdn_fla_scratch`, sub-divided). Strides match the packed conv layout
/// (qk_stride=v_stride=conv_dim, gb_stride=2*nv) exactly like the wy4/chunk64 path.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_fla(
    gpu: &dyn GpuBackend,
    k_recompute_wu: KernelHandle,
    k_chunk_delta_h: KernelHandle,
    // wmma + DV-block-split spine (gated_delta_rule_chunk_delta_h_tc_vblock). When
    // non-zero AND ATLAS_GDN_TC_VBLOCK=1, replaces the scalar ksplit spine (drop-in
    // ABI; grid y = batch·num_dv_blocks, smem 81KB vs 97KB). KernelHandle(0) = off.
    k_chunk_delta_h_tc_vblock: KernelHandle,
    k_chunk_fwd_o: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    w_out: DevicePtr,
    u_out: DevicePtr,
    s_out: DevicePtr,
    uc_out: DevicePtr,
    gc_out: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_chunks: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    // h_state passed as a device POINTER TABLE (one [nv,kd,vd] per request) when
    // batched co-dispatch reuses the per-request states; false = contiguous base.
    h_state_is_table: bool,
    // VARLEN (ragged co-dispatch): per-stream cu_seqlens (token offsets, batch+1
    // ints) + cu_chunks (chunk offsets, batch+1 ints) on device. When is_varlen,
    // `num_chunks` must be the MAX over streams (grid x). is_varlen=false →
    // cu_* unused (pass NULL).
    cu_seqlens: DevicePtr,
    cu_chunks: DevicePtr,
    is_varlen: bool,
    profile: bool,
    stream: u64,
) -> Result<()> {
    const C: u32 = 64; // CHUNK (kernel constant)
    let (kd, vd) = (k_dim, v_dim);
    // smem byte sizes — identical formulas to the GATE-B example (validated).
    let smem_wu = C * kd * 2 + C * C * 4 + C * C * 4 + C * 4;
    let smem_dh = 2 * (C * (2 * kd + vd) * 2) + 2 * C * 4 + 2 * (C + 1) * 4;
    let smem_fo = C * kd * 2 + C * kd * 2 + C * C * 4 + C * vd * 2 + kd * vd * 2 + 2 * C * 4;

    let mut t0: Option<std::time::Instant> = if profile {
        gpu.synchronize(stream)?;
        Some(std::time::Instant::now())
    } else {
        None
    };

    macro_rules! prof {
        ($label:expr, $t0:expr) => {
            if let Some(t0) = $t0.take() {
                gpu.synchronize(stream)?;
                let elapsed = t0.elapsed().as_micros();
                tracing::info!("  SSM prefill [{}] N={}: {}µs", $label, seq_len, elapsed);
                *$t0 = Some(std::time::Instant::now());
            }
        };
    }

    // Kernel 1: recompute_wu.
    KernelLaunch::new(gpu, k_recompute_wu)
        .grid([num_chunks, num_v_heads, batch_size])
        .block([128, 1, 1])
        .shared_mem(smem_wu)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(w_out)
        .arg_ptr(u_out)
        .arg_ptr(gc_out)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_chunks)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(kd)
        .arg_u32(vd)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_ptr(cu_seqlens)
        .arg_ptr(cu_chunks)
        .arg_u32(is_varlen as u32)
        .launch(stream)?;
    prof!("gdn_fla_recompute_wu", &mut t0);

    // Kernel 2: chunk_delta_h — scalar ksplit OR the wmma + DV-block-split tc_vblock
    // (gated). tc_vblock is a drop-in ABI; only the grid-y extent (batch·num_dv_blocks)
    // and the dynamic smem differ. The DV axis is never a contraction axis so the
    // per-DV-block slices are independent → bit-parity with ksplit (validated isolated).
    let use_tcvb = k_chunk_delta_h_tc_vblock.0 != 0
        && std::env::var("ATLAS_GDN_TC_VBLOCK").ok().as_deref() == Some("1");
    const DV_BLK: u32 = 64; // matches the kernel's compile-time DV_BLK
    let num_dv_blk = (vd / DV_BLK).max(1); // 2 for Holo (vd=128)
    // tc_vblock smem: St[DV_BLK*kd] + ws[C*DV_BLK]f32 + buf[2][C*kd + C*DV_BLK] + gcb + decb
    let smem_tcvb = DV_BLK * kd * 2
        + C * DV_BLK * 4
        + 2 * (C * kd + C * DV_BLK) * 2
        + 2 * C * 4
        + 2 * (C + 1) * 4;
    let (k_cdh, cdh_grid_y, cdh_smem) = if use_tcvb {
        (
            k_chunk_delta_h_tc_vblock,
            batch_size * num_dv_blk,
            smem_tcvb,
        )
    } else {
        (k_chunk_delta_h, batch_size, smem_dh)
    };
    KernelLaunch::new(gpu, k_cdh)
        .grid([num_v_heads, cdh_grid_y, 1])
        .block([256, 1, 1])
        .shared_mem(cdh_smem)
        .arg_ptr(h_state)
        .arg_ptr(w_out)
        .arg_ptr(u_out)
        .arg_ptr(key)
        .arg_ptr(gate)
        .arg_ptr(gc_out)
        .arg_ptr(s_out)
        .arg_ptr(uc_out)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_chunks)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(kd)
        .arg_u32(vd)
        .arg_u32(qk_stride)
        .arg_u32(gb_stride)
        .arg_u32(h_state_is_table as u32)
        .arg_ptr(cu_seqlens)
        .arg_ptr(cu_chunks)
        .arg_u32(is_varlen as u32)
        .launch(stream)?;
    prof!("gdn_fla_chunk_delta_h", &mut t0);

    // Kernel 3: chunk_fwd_o.
    KernelLaunch::new(gpu, k_chunk_fwd_o)
        .grid([num_chunks, num_v_heads, batch_size])
        .block([128, 1, 1])
        .shared_mem(smem_fo)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(gate)
        .arg_ptr(gc_out)
        .arg_ptr(s_out)
        .arg_ptr(uc_out)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_chunks)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(kd)
        .arg_u32(vd)
        .arg_u32(qk_stride)
        .arg_u32(gb_stride)
        .arg_ptr(cu_seqlens)
        .arg_ptr(cu_chunks)
        .arg_u32(is_varlen as u32)
        .launch(stream)?;
    if let Some(t0) = t0 {
        gpu.synchronize(stream)?;
        let elapsed = t0.elapsed().as_micros();
        tracing::info!(
            "  SSM prefill [gdn_fla_chunk_fwd_o] N={}: {}µs",
            seq_len,
            elapsed
        );
    }
    Ok(())
}
