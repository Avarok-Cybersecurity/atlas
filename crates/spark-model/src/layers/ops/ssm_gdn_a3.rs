// SPDX-License-Identifier: AGPL-3.0-only

//! GDN FLA prefill op — extracted from `ssm_gdn_a2.rs` during the ≤500-line
//! split (the 3-kernel FLA path grew past the cap when the vtile spine added
//! its handle + dispatch). All public items remain available at
//! `crate::layers::ops::*` via the re-export in `ops.rs`.
#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

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
    k_chunk_delta_h_vtile: KernelHandle,
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
    // L aliases the kk Gram (disjoint triangles) — one C*C*4 buffer, not two.
    let smem_wu = C * kd * 2 + C * C * 4 + C * 4;
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
        .block([256, 1, 1])
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
    // vtile is the DEFAULT spine when it resolved: 2.15-2.18x over ksplit, bit-exact
    // to cos=1.0000. `ATLAS_GDN_VTILE=0` is the kill switch back to ksplit.
    // vtile is OPT-IN (`ATLAS_GDN_VTILE=1`), NOT the default, despite being
    // 2.15-2.18x over ksplit with cos=1.0000 on the isolated tensors.
    //
    // It regresses tool-calling accuracy below the gate floors on BOTH models,
    // measured by a full record campaign:
    //   bfcl-subset (27B)  83.62 / 82.72  vs floors 83.42 / 83.32  FAIL
    //   bfcl-echolp (35B)  85.96 / 86.09  vs floors 86.10 / 86.50  FAIL
    //   same binary, ATLAS_GDN_VTILE=0    84.22 / 84.12            PASS
    // The bisect is one-variable and decisive: disabling only the spine restores
    // the historical numbers EXACTLY, which also clears chunk_fwd_o, the
    // recompute_wu pass split and the kk/L alias — all three were active in the
    // passing run.
    //
    // The cause is structural, not a bug. vtile reaches 512 threads only via
    // SPLIT=4 (threads = (V_DIM/VT)*SPLIT, V_DIM=128), and SPLIT=4 reduces k in
    // FOUR partial sums through a 2-round shfl butterfly where ksplit uses two.
    // That reassociates the f32 accumulation of <W_i, S[:,v]> on every token of
    // every chunk. At SPLIT=2 the design caps at 256 threads — which IS ksplit.
    // The warp-density win and the accumulation order are therefore inseparable
    // as written: recovering it needs an order-PRESERVING way to add warps.
    //
    // ★ Neither cos>=0.99 on the isolated spine NOR a byte-identical greedy
    // comparison on a single prompt caught this. A drift too small to change one
    // trajectory still moved BFCL by 1.4 points across 995 samples.
    let use_vtile = k_chunk_delta_h_vtile.0 != 0
        && std::env::var("ATLAS_GDN_VTILE").ok().as_deref() == Some("1");
    let use_tcvb = !use_vtile
        && k_chunk_delta_h_tc_vblock.0 != 0
        && std::env::var("ATLAS_GDN_TC_VBLOCK").ok().as_deref() == Some("1");
    const DV_BLK: u32 = 64; // matches the kernel's compile-time DV_BLK
    let num_dv_blk = (vd / DV_BLK).max(1); // 2 for Holo (vd=128)
    // tc_vblock smem: St[DV_BLK*kd] + ws[C*DV_BLK]f32 + buf[2][C*kd + C*DV_BLK] + gcb + decb
    let smem_tcvb = DV_BLK * kd * 2
        + C * DV_BLK * 4
        + 2 * (C * kd + C * DV_BLK) * 2
        + 2 * C * 4
        + 2 * (C + 1) * 4;
    // vtile stages {W,K,U} single-buffered plus one decay row; it does NOT split
    // the DV axis, so grid.y stays `batch_size`.
    let smem_vtile = C * kd * 2 + C * kd * 2 + C * vd * 2 + (C + 1) * 4;
    let (k_cdh, cdh_grid_y, cdh_smem, cdh_block) = if use_vtile {
        (k_chunk_delta_h_vtile, batch_size, smem_vtile, 512u32)
    } else if use_tcvb {
        (
            k_chunk_delta_h_tc_vblock,
            batch_size * num_dv_blk,
            smem_tcvb,
            256u32,
        )
    } else {
        (k_chunk_delta_h, batch_size, smem_dh, 256u32)
    };
    KernelLaunch::new(gpu, k_cdh)
        .grid([num_v_heads, cdh_grid_y, 1])
        .block([cdh_block, 1, 1])
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
        .block([512, 1, 1])
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
