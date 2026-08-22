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
    k_chunk_delta_h_fused: KernelHandle,
    k_chunk_delta_h_tma: KernelHandle,
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

    // Kernel 2: chunk_delta_h — the fused spine OR the wmma + DV-block-split
    // tc_vblock (gated). Both are drop-in ABI; only grid-y extent, block size and
    // dynamic smem differ.
    //
    // DEFAULT is `..._vfused` (SPLIT=2, 256 threads): the two per-chunk passes are
    // folded into one, which collapses `duc` from a CHUNK-long array to a scalar and
    // drops smem 99,336 -> 49,412 B. That is worth 2.01x over ksplit on the isolated
    // spine and ~68 ms of cold TTFT (one-variable A/B, same binary, 10 reps/leg).
    //
    // `ATLAS_GDN_VTILE=1` raises the same core to SPLIT=4 / 512 threads for 2.15x.
    // It is NOT the default: it regressed tool-calling accuracy below the gate
    // floors on BOTH models in a full record campaign —
    //   bfcl-subset (27B)  83.62 / 82.72  vs floors 83.42 / 83.32  FAIL
    //   bfcl-echolp (35B)  85.96 / 86.09  vs floors 86.10 / 86.50  FAIL
    //   same binary, spine off             84.22 / 84.12            PASS
    //
    // ★ WHAT IS AND IS NOT KNOWN. The ssm-poisoning gate is the 4-minute tripwire
    // that separates these, and it bisects the two changes vtile made at once:
    //     ksplit  (unfused, SPLIT=2)  12/12 replays byte-identical
    //     vfused  (fused,   SPLIT=2)  12/12   <- shipped
    //     vtile   (fused,   SPLIT=4)   1/12
    // So the FUSION is innocent and SPLIT=4 is implicated. The mechanism is NOT
    // reassociation of the k-sum, which an earlier revision of this comment claimed:
    // accumulating the SPLIT-way butterfly fold in Neumaier-compensated form scored
    // 0/12 — no better — so that hypothesis is refuted and the true cause of
    // SPLIT=4's drift is UNKNOWN. Since SPLIT=4 buys only 7% over SPLIT=2, the
    // warp-density half was never where the win was.
    //
    // ★ Neither cos>=0.99 on the isolated spine NOR a byte-identical greedy
    // comparison on a single prompt caught this. A drift too small to change one
    // trajectory still moved BFCL by 1.4 points across 995 samples. Use the
    // ssm-poisoning tripwire before trusting any change to this kernel.
    let use_fused = k_chunk_delta_h_fused.0 != 0
        && std::env::var("ATLAS_GDN_VTILE").ok().as_deref() != Some("0");
    let use_tcvb = !use_fused
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
    // The fused spine stages {W,K,U} single-buffered plus one decay row; it does NOT
    // split the DV axis, so grid.y stays `batch_size`. smem is identical for both
    // members — only the thread count differs, and it must match the kernel that
    // init.rs actually loaded for the same env value.
    // `..._pipe` double-buffers {W,K,U} through `cp.async`, so it needs the SAME
    // footprint the original (also double-buffered) spine uses — `smem_dh`. Under-
    // sizing this reads the second slot out of bounds, so the selector has to agree
    // with the kernel `init.rs` loaded for the same env value.
    let pipe = std::env::var("ATLAS_GDN_PIPE").ok().as_deref() == Some("1");
    let smem_fused = if pipe {
        smem_dh
    } else {
        C * kd * 2 + C * kd * 2 + C * vd * 2 + (C + 1) * 4
    };
    let fused_block = match std::env::var("ATLAS_GDN_VTILE").ok().as_deref() {
        Some("1") if !pipe => 512u32, // SPLIT=4 build
        _ => 256u32,                  // SPLIT=2 build (default, and the pipe build)
    };
    // ── TMA path (ATLAS_GDN_TMA=1) ───────────────────────────────────────────
    // Every precondition is CHECKED, not assumed. The descriptors are encoded
    // from the compile-time tile (K_DIM/V_DIM = 128, CHUNK = 64), so a runtime
    // head narrower than the tile would load the wrong columns SILENTLY — TMA
    // reports no error for a well-formed descriptor pointed at the wrong shape.
    // Varlen is excluded because `choff` then comes from `cu_chunks` and the
    // flat row count the descriptor needs is not known on the host.
    let tma_requested = std::env::var("ATLAS_GDN_TMA").ok().as_deref() == Some("1");
    // ★ NAME THE GUARD THAT REJECTED. A perf path that asks to be enabled and
    // silently is not measures as "no effect" — PR #296 shipped exactly that
    // (an ldmatrix GEMM that fell back with no error while both gates stayed
    // green), and this path reproduced it during bring-up: an A/B ran with the
    // env set, fell back to `vfused`, and the two arms differed by noise.
    let tma_reject: Option<&str> = if !tma_requested {
        Some("not requested")
    } else if k_chunk_delta_h_tma.0 == 0 {
        Some("kernel absent from this image")
    } else if is_varlen {
        Some(
            "varlen: choff comes from cu_chunks, so the descriptor's flat row count is unknown host-side",
        )
    } else if kd != 128 || vd != 128 || C != 64 {
        Some("head/chunk differs from the compile-time tile the descriptors encode")
    } else if !qk_stride.is_multiple_of(8) {
        Some("qk_stride is not a multiple of 8 (bf16 row pitch must be 16-byte aligned)")
    } else {
        None
    };
    if tma_requested && let Some(why) = tma_reject {
        tracing::warn!("ATLAS_GDN_TMA=1 but the TMA spine is NOT running: {why}");
    }
    let tma_ok = tma_reject.is_none();
    if tma_ok {
        tracing::info!("GDN state spine: gated_delta_rule_chunk_delta_h_tma");
    }
    // `cuda_backend` (and with it `TensorMap`) only exists under the cuda
    // feature; the metal build has no TMA and must not reference it. The guard
    // above already resolves to false there via the absent kernel handle, but a
    // `use` is resolved at compile time regardless of the branch being taken.
    #[cfg(feature = "cuda")]
    if tma_ok {
        use spark_runtime::cuda_backend::tensormap::TensorMap;
        // W/U are [total_blocks][CHUNK][tile] flattened; as a 2-D tensor that is
        // (total_blocks * CHUNK) rows of `tile` columns, contiguous.
        let blocks = (batch_size as u64) * (num_chunks as u64) * (num_v_heads as u64);
        let w_map =
            TensorMap::tiled_2d_bf16(w_out, blocks * C as u64, kd as u64, kd as u64, C, kd)?;
        let u_map =
            TensorMap::tiled_2d_bf16(u_out, blocks * C as u64, vd as u64, vd as u64, C, vd)?;
        // K is a VIEW into the packed qkvz tensor: rows are tokens at a
        // `qk_stride` pitch, and the kernel supplies `kh * K_DIM` as the column
        // origin. This is the gather `cdh_prefetch` does one row at a time.
        let k_map = TensorMap::tiled_2d_bf16(
            key,
            (batch_size as u64) * (seq_len as u64),
            qk_stride as u64,
            qk_stride as u64,
            C,
            kd,
        )?;
        KernelLaunch::new(gpu, k_chunk_delta_h_tma)
            .grid([num_v_heads, batch_size, 1])
            .block([256, 1, 1])
            .shared_mem(smem_dh)
            .arg_ptr(h_state)
            .arg_tensormap(w_map.bytes())
            .arg_tensormap(u_map.bytes())
            .arg_tensormap(k_map.bytes())
            .arg_ptr(gc_out)
            .arg_ptr(s_out)
            .arg_ptr(uc_out)
            .arg_u32(batch_size)
            .arg_u32(seq_len)
            .arg_u32(num_chunks)
            .arg_u32(num_k_heads)
            .arg_u32(num_v_heads)
            .arg_u32(vd)
            .arg_u32(h_state_is_table as u32)
            .arg_ptr(cu_seqlens)
            .arg_ptr(cu_chunks)
            .arg_u32(is_varlen as u32)
            .launch(stream)?;
        prof!("gdn_fla_chunk_delta_h", &mut t0);
    }

    // Kernel 2 (non-TMA). Both paths write s_out/uc_out and fall through to
    // kernel 3, which is identical either way.
    if !tma_ok {
        let (k_cdh, cdh_grid_y, cdh_smem, cdh_block) = if use_fused {
            (k_chunk_delta_h_fused, batch_size, smem_fused, fused_block)
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
    }

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
