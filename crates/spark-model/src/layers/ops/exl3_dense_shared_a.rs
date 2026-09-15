// SPDX-License-Identifier: AGPL-3.0-only

//! The shared-A dense launch body, split out of `exl3_dense.rs` to keep it
//! under the 500-line cap.

use super::*;

/// [`exl3_dense_linear_shared_a`] with the fused-egress arm chosen
/// explicitly (the public entry reads the kill switch once per process; the
/// launch-plan tests pin BOTH plans through this).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_linear_shared_a(
    gpu: &dyn GpuBackend,
    ws: &[(Exl3DenseWeight, Exl3DenseOut)],
    a_bf16: DevicePtr,
    m: usize,
    stage: &Exl3DenseStage,
    stream: u64,
    fused_egress: bool,
) -> Result<()> {
    ensure!(!ws.is_empty(), "exl3_dense_linear: no weights");
    ensure!(m >= 1, "exl3_dense_linear: m == 0");
    let k = ws[0].0.in_dim;
    ensure!(
        k <= stage.max_in,
        "exl3_dense_linear: in_dim {k} exceeds the stage's max_in {}",
        stage.max_in
    );
    for (w, out) in ws {
        ensure!(
            w.in_dim == k,
            "exl3_dense_linear: shared-A weights disagree on in_dim ({} vs {k})",
            w.in_dim
        );
        ensure!(
            w.out_dim <= stage.max_out,
            "exl3_dense_linear: out_dim {} exceeds the stage's max_out {}",
            w.out_dim,
            stage.max_out
        );
        if let Some(ld) = out.ld {
            ensure!(
                ld >= w.out_dim,
                "exl3_dense_linear: destination row stride {ld} < out_dim {}",
                w.out_dim
            );
        }
        ensure!(!out.ptr.is_null(), "exl3_dense_linear: null destination");
        ensure!(
            !out.fp32 || stage.c_f32_elems >= stage.rows_cap.min(m) * w.out_dim,
            "exl3_dense_linear: fp32-C destination for out_dim {} needs {} f32 elems of \
             stage.c_f32 (have {}) — size the stage's max_out_f32 for this projection",
            w.out_dim,
            stage.rows_cap.min(m) * w.out_dim,
            stage.c_f32_elems
        );
    }
    let _section = stage.launch.section(gpu, stream)?;
    let launch = &*stage.launch;

    if m <= EXL3_GEMV_MAX_M {
        // The f16 ingress launch is needed only when some weight in the group
        // takes the GEMV tier (K in 2..=4). K in {5,6,8} goes to the f32-C
        // GEMM's `_abf16` twin, which converts BF16 -> f16 inside its
        // input-Hadamard prologue (bit-identical to convert-then-GEMM) — one
        // launch fewer per projection group on the decode path.
        let group_needs_f16 = ws.iter().any(|(w, _)| exl3_gemv_serves_k(w.k_bits));
        if group_needs_f16 {
            exl3_bf16_to_f16(gpu, a_bf16, stage.a_f16, m * k, stream)?;
        }
        for (w, out) in ws {
            let n = w.out_dim;
            if !exl3_gemv_serves_k(w.k_bits) {
                if fused_egress {
                    // One launch: BF16 ingress in the prologue, BF16(C) into
                    // the (contiguous or pitched) destination from the
                    // epilogue. Same bytes as the bracketed form below.
                    exl3_gemm_abf16_obf16(
                        gpu,
                        a_bf16,
                        w.trellis,
                        stage.c_f32,
                        out.ptr,
                        out.ld.unwrap_or(n),
                        m,
                        k,
                        n,
                        w.k_bits,
                        w.cb,
                        launch.locks,
                        w.suh,
                        stage.a_had_f16,
                        w.svh,
                        None,
                        launch.sm_count,
                        stream,
                    )?;
                    continue;
                }
                exl3_gemm_abf16(
                    gpu,
                    a_bf16,
                    w.trellis,
                    stage.c_f32,
                    m,
                    k,
                    n,
                    w.k_bits,
                    w.cb,
                    launch.locks,
                    w.suh,
                    stage.a_had_f16,
                    w.svh,
                    None,
                    launch.sm_count,
                    stream,
                )?;
                match out.ld {
                    Some(ld) if ld != n => {
                        exl3_f32_to_bf16_2d(gpu, stage.c_f32, out.ptr, m, n, n, ld, stream)?
                    }
                    _ => exl3_f32_to_bf16(gpu, stage.c_f32, out.ptr, m * n, stream)?,
                }
                continue;
            }
            // The GEMV tier exists for K in 2..=4 only. Not an error when the
            // heuristic declines: every cooperative launch runs under this
            // call's section, so the split-K GEMM at small m is as safe as
            // the GEMV here.
            let launched = exl3_gemv_serves_k(w.k_bits)
                && exl3_gemv(
                    gpu,
                    stage.a_f16,
                    w.trellis,
                    stage.c_f32,
                    m,
                    k,
                    n,
                    w.k_bits,
                    w.cb,
                    true,
                    launch.locks,
                    w.suh,
                    stage.a_had_f16,
                    w.svh,
                    None,
                    launch.sm_count,
                    stream,
                )?;
            if !launched {
                exl3_gemm(
                    gpu,
                    stage.a_f16,
                    w.trellis,
                    stage.c_f32,
                    m,
                    k,
                    n,
                    w.k_bits,
                    w.cb,
                    true,
                    launch.locks,
                    w.suh,
                    stage.a_had_f16,
                    w.svh,
                    None,
                    launch.sm_count,
                    stream,
                )?;
            }
            match out.ld {
                Some(ld) if ld != n => {
                    exl3_f32_to_bf16_2d(gpu, stage.c_f32, out.ptr, m, n, n, ld, stream)?
                }
                _ => exl3_f32_to_bf16(gpu, stage.c_f32, out.ptr, m * n, stream)?,
            }
        }
        return Ok(());
    }

    // Reconstruct-to-BF16 prefill tier (opt-in, `ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS`):
    // one trellis decode per weight per call + a fixed-config BF16 GEMM. Not
    // bit-identical to the trellis GEMM below — see `exl3_dense/reconstruct.rs`.
    if let Some(rs) = stage.recon.as_ref().filter(|rs| rs.takes(m)) {
        return reconstruct::run_reconstruct_tier(gpu, ws, a_bf16, m, stage, rs, stream);
    }

    // GEMM tier, row-batched at the slab capacity.
    let mut r0 = 0usize;
    while r0 < m {
        let rows = (m - r0).min(stage.rows_cap);
        exl3_bf16_to_f16(
            gpu,
            a_bf16.offset(r0 * k * 2),
            stage.a_f16,
            rows * k,
            stream,
        )?;
        for (w, out) in ws {
            let n = w.out_dim;
            let ld = out.ld.unwrap_or(n);
            let dst = out.ptr.offset(r0 * ld * 2);
            // fp32 C (residual-bound projections): accumulate in the f32 slab,
            // egress with the f32 converter. Otherwise contiguous fp16 C lands
            // in the BF16 destination's own bytes (same 2 B/elem), then
            // converts in place (each index read-then-written once); strided
            // fp16 C stages through c_f16.
            let c = if out.fp32 {
                stage.c_f32
            } else if ld == n {
                dst
            } else {
                stage.c_f16
            };
            exl3_gemm(
                gpu,
                stage.a_f16,
                w.trellis,
                c,
                rows,
                k,
                n,
                w.k_bits,
                w.cb,
                out.fp32,
                launch.locks,
                w.suh,
                stage.a_had_f16,
                w.svh,
                None,
                launch.sm_count,
                stream,
            )?;
            match (out.fp32, ld == n) {
                (true, true) => exl3_f32_to_bf16(gpu, stage.c_f32, dst, rows * n, stream)?,
                (true, false) => {
                    exl3_f32_to_bf16_2d(gpu, stage.c_f32, dst, rows, n, n, ld, stream)?
                }
                (false, true) => exl3_f16_to_bf16(gpu, dst, dst, rows * n, stream)?,
                (false, false) => {
                    exl3_f16_to_bf16_2d(gpu, stage.c_f16, dst, rows, n, n, ld, stream)?
                }
            }
        }
        r0 += rows;
    }
    Ok(())
}
