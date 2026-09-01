// SPDX-License-Identifier: AGPL-3.0-only

//! Native dense-linear dispatch over a packed EXL3 (QTIP trellis) weight —
//! the reusable arm behind `ATLAS_EXL3_NATIVE_DENSE=1` for qwen4_exp's GDN
//! (`in_proj_qkv`, `in_proj_z`, `out_proj`) and attention (`q/k/v/o_proj`)
//! projections. No new matmul kernels: everything routes through the proven
//! [`exl3_gemv`] / [`exl3_gemm`] wrappers; this module owns the BF16
//! boundary, the row batching, and the concurrency contract.
//!
//! Data path per call (all stream-ordered, no host sync, no allocation):
//!
//! ```text
//!   A bf16 [m, k] --exl3_bf16_to_f16--> stage.a_f16 (raw fp16)
//!   m <= 8 : exl3_gemv (fp32 C into stage.c_f32; gemm fallthrough when the
//!            GEMV heuristic declines) --exl3_f32_to_bf16[_2d]--> dst
//!   m >  8 : per row batch of stage.rows_cap:
//!              contiguous dst: exl3_gemm fp16 C straight into dst, then
//!                              exl3_f16_to_bf16 IN PLACE (lm_head precedent)
//!              strided dst:    exl3_gemm fp16 C into stage.c_f16, then
//!                              exl3_f16_to_bf16_2d into the arena rows
//! ```
//!
//! `A_had` (the kernels' rotation scratch) is a SEPARATE slab, never aliased
//! to A here: a shared-A group ([`exl3_dense_linear_shared_a`], GDN qkv+z or
//! attention q/k/v) ingresses once and lets each weight rotate the same raw A
//! under its own `suh`.
//!
//! Concurrency: one [`Exl3LaunchState::section`] wraps the whole call (host
//! mutex + device fence — see `exl3_dense/launch_state.rs`), shared with the
//! MoE arm. Cooperative launches are not graph-capturable: the calling arm
//! must sit behind the model's `exl3_graph_veto` and refuse `graph_capture`.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::exl3::{Exl3Codebook, Exl3Weight};

// Explicit `#[path]`: `ops.rs` loads THIS file with one too (the exl3_matmul
// precedent).
#[path = "exl3_dense/launch_state.rs"]
mod launch_state;
pub use launch_state::{Exl3LaunchState, Exl3Section};
#[path = "exl3_dense/stage.rs"]
mod stage;
pub use stage::{EXL3_DENSE_STAGE_ROWS_DEFAULT, Exl3DenseStage};

use super::exl3_matmul::{
    EXL3_GEMV_MAX_M, exl3_bf16_to_f16, exl3_f16_to_bf16, exl3_f16_to_bf16_2d, exl3_f32_to_bf16,
    exl3_f32_to_bf16_2d, exl3_gemm, exl3_gemv,
};

/// One packed dense linear as the kernels address it: device pointers +
/// geometry + the KERNEL codebook index (1 = MCG, 2 = MUL1). Built from the
/// loader's [`Exl3Weight`] via [`Exl3DenseWeight::from_exl3`]; `Copy` so a
/// layer can carry it inline like the other weight descriptors.
#[derive(Debug, Clone, Copy)]
pub struct Exl3DenseWeight {
    /// u16 trellis `[in/16, out/16, 16K]`.
    pub trellis: DevicePtr,
    /// f16 `[in_dim]` input Hadamard signs.
    pub suh: DevicePtr,
    /// f16 `[out_dim]` output Hadamard signs.
    pub svh: DevicePtr,
    pub in_dim: usize,
    pub out_dim: usize,
    pub k_bits: u32,
    /// Kernel codebook index: 1 = MCG, 2 = MUL1.
    pub cb: u32,
}

impl Exl3DenseWeight {
    /// Validate against the compiled dense envelope (K in 2..=4 so the m<=8
    /// GEMV tier exists, cb MCG/MUL1, dims % 128) and convert the codebook.
    /// The stricter serving policy (`exl3_native_supported`, K in {2,4}) is
    /// the loader's predicate, applied before this.
    pub fn from_exl3(w: &Exl3Weight) -> Result<Self> {
        let cb = match w.cb {
            Exl3Codebook::Mcg => 1,
            Exl3Codebook::Mul1 => 2,
            Exl3Codebook::Inst3 => {
                anyhow::bail!("EXL3 dense: codebook 3INST has no compiled kernel instances")
            }
        };
        ensure!(
            (2..=4).contains(&w.k_bits),
            "EXL3 dense: K={} is outside the GEMV envelope (2..=4)",
            w.k_bits
        );
        ensure!(
            w.in_dim >= 128
                && w.in_dim.is_multiple_of(128)
                && w.out_dim >= 128
                && w.out_dim.is_multiple_of(128),
            "EXL3 dense: [{} -> {}] dims must be multiples of 128",
            w.in_dim,
            w.out_dim
        );
        ensure!(
            !w.trellis.is_null() && !w.suh.is_null() && !w.svh.is_null(),
            "EXL3 dense: null trellis/suh/svh pointer"
        );
        Ok(Self {
            trellis: w.trellis,
            suh: w.suh,
            svh: w.svh,
            in_dim: w.in_dim,
            out_dim: w.out_dim,
            k_bits: w.k_bits,
            cb,
        })
    }

    /// Resident packed bytes (trellis + suh + svh + codebook flag) — the
    /// same accounting as `Exl3Weight::packed_bytes`, for the load log.
    pub fn packed_bytes(&self) -> usize {
        self.in_dim * self.out_dim * self.k_bits as usize / 8 + (self.in_dim + self.out_dim) * 2 + 4
    }

    /// What the same linear costs as dense BF16 `[out, in]`.
    pub fn bf16_bytes(&self) -> usize {
        self.in_dim * self.out_dim * 2
    }
}

impl TryFrom<&Exl3Weight> for Exl3DenseWeight {
    type Error = anyhow::Error;
    fn try_from(w: &Exl3Weight) -> Result<Self> {
        Self::from_exl3(w)
    }
}

impl TryFrom<Exl3Weight> for Exl3DenseWeight {
    type Error = anyhow::Error;
    fn try_from(w: Exl3Weight) -> Result<Self> {
        Self::from_exl3(&w)
    }
}

/// Destination of one projection: BF16 rows starting at `ptr`, either
/// contiguous (`ld == None`, row stride = `out_dim`) or pitched into a wider
/// arena row (`ld = Some(row stride in ELEMENTS)`, e.g. the GDN `[Q|K|V|Z]`
/// row of 16384 with `ptr` offset to the block's first column).
#[derive(Debug, Clone, Copy)]
pub struct Exl3DenseOut {
    pub ptr: DevicePtr,
    pub ld: Option<usize>,
    /// GEMM tier (m > 8) accumulates and stores C in fp32 (split-K partials
    /// included) before the BF16 egress, instead of fp16 C converted in
    /// place. Upstream pins the GDN / attention BLOCK outputs (out_proj,
    /// o_proj — the projections that feed the residual stream) to fp32
    /// `out_dtype`; fp16 C there would be a precision/range seam between the
    /// m<=8 tier (always fp32 C) and prefill, and a rotated split-K partial
    /// can exceed the final output by up to sqrt(128)x. Costs the stage's
    /// `rows_cap x max_out_f32` f32 slab and an f32->bf16 pass.
    pub fp32: bool,
}

impl Exl3DenseOut {
    pub fn contiguous(ptr: DevicePtr) -> Self {
        Self {
            ptr,
            ld: None,
            fp32: false,
        }
    }
    pub fn strided(ptr: DevicePtr, ld_elems: usize) -> Self {
        Self {
            ptr,
            ld: Some(ld_elems),
            fp32: false,
        }
    }
    /// Request fp32 C on the GEMM tier (see the field doc).
    pub fn with_fp32(self) -> Self {
        Self { fp32: true, ..self }
    }
}

/// `dst = A @ W` for ONE packed weight. `a_bf16`: BF16 `[m, in_dim]`
/// contiguous; `out`: BF16 `[m, out_dim]` contiguous or pitched. Any `m`.
pub fn exl3_dense_linear(
    gpu: &dyn GpuBackend,
    w: &Exl3DenseWeight,
    a_bf16: DevicePtr,
    out: Exl3DenseOut,
    m: usize,
    stage: &Exl3DenseStage,
    stream: u64,
) -> Result<()> {
    exl3_dense_linear_shared_a(gpu, &[(*w, out)], a_bf16, m, stage, stream)
}

/// Several packed weights over the SAME activation (GDN `in_proj_qkv` +
/// `in_proj_z`; attention `q/k/v`): ingress once, one matmul per weight,
/// all under ONE dispatch section. Every weight must share `in_dim`.
pub fn exl3_dense_linear_shared_a(
    gpu: &dyn GpuBackend,
    ws: &[(Exl3DenseWeight, Exl3DenseOut)],
    a_bf16: DevicePtr,
    m: usize,
    stage: &Exl3DenseStage,
    stream: u64,
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
        exl3_bf16_to_f16(gpu, a_bf16, stage.a_f16, m * k, stream)?;
        for (w, out) in ws {
            let n = w.out_dim;
            let launched = exl3_gemv(
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

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn weight(gpu: &MockGpuBackend, k: usize, n: usize) -> Exl3DenseWeight {
        Exl3DenseWeight {
            trellis: gpu.alloc((k / 16) * (n / 16) * 16 * 4 * 2).unwrap(),
            suh: gpu.alloc(k * 2).unwrap(),
            svh: gpu.alloc(n * 2).unwrap(),
            in_dim: k,
            out_dim: n,
            k_bits: 4,
            cb: 2,
        }
    }

    fn kernel_names(gpu: &MockGpuBackend) -> Vec<String> {
        gpu.kernel_lookups_snapshot()
            .into_iter()
            .map(|(_, f)| f)
            .collect()
    }

    fn contiguous(dst: DevicePtr) -> Exl3DenseOut {
        Exl3DenseOut::contiguous(dst)
    }

    #[test]
    fn launch_plan_gemv_tier_and_row_batched_gemm_tier() {
        // The mock records kernel resolutions, which IS the launch plan:
        // ingress / matmul / egress per tier, and the row batching above the
        // stage capacity (numerics are the GPU parity example's job).
        let gpu = MockGpuBackend::new();
        let launch = std::sync::Arc::new(Exl3LaunchState::new(&gpu).unwrap());
        // 256-row stage (the env override would break the batching
        // assertion — the test never sets it).
        let stage = Exl3DenseStage::new(&gpu, launch, 256, 6144, 12288).unwrap();
        let w = weight(&gpu, 2560, 10240);
        let a = gpu.alloc(700 * 2560 * 2).unwrap();
        let dst = gpu.alloc(700 * 10240 * 2).unwrap();

        let before = gpu.kernel_lookups_snapshot().len();
        exl3_dense_linear(&gpu, &w, a, contiguous(dst), 1, &stage, 0).unwrap();
        let names = kernel_names(&gpu)[before..].to_vec();
        assert_eq!(
            names,
            vec![
                "exl3_bf16_to_f16",
                "exl3_gemv_k4_cb2_m0_cfg1_f32",
                "exl3_f32_to_bf16"
            ],
            "m=1 must take the f32-C GEMV tier"
        );

        let before = gpu.kernel_lookups_snapshot().len();
        exl3_dense_linear(&gpu, &w, a, contiguous(dst), 700, &stage, 0).unwrap();
        let names = kernel_names(&gpu)[before..].to_vec();
        // 3 batches (256 + 256 + 188) x (ingress, gemm, in-place convert).
        assert_eq!(names.len(), 9, "{names:?}");
        for b in 0..3 {
            assert_eq!(names[b * 3], "exl3_bf16_to_f16");
            assert!(names[b * 3 + 1].starts_with("exl3_gemm_k4_cb2_sh"));
            assert!(names[b * 3 + 1].ends_with("_f16"));
            assert_eq!(names[b * 3 + 2], "exl3_f16_to_bf16");
        }

        // Strided destination: staged f16 C + the 2-D egress; a shared-A
        // pair ingresses ONCE per row batch.
        let z = weight(&gpu, 2560, 6144);
        let arena = gpu.alloc(700 * 16384 * 2).unwrap();
        let before = gpu.kernel_lookups_snapshot().len();
        exl3_dense_linear_shared_a(
            &gpu,
            &[
                (w, Exl3DenseOut::strided(arena, 16384)),
                (z, Exl3DenseOut::strided(arena.offset(10240 * 2), 16384)),
            ],
            a,
            300,
            &stage,
            0,
        )
        .unwrap();
        let names = kernel_names(&gpu)[before..].to_vec();
        assert_eq!(
            names.iter().filter(|n| *n == "exl3_bf16_to_f16").count(),
            2,
            "one ingress per row batch: {names:?}"
        );
        assert_eq!(
            names.iter().filter(|n| *n == "exl3_f16_to_bf16_2d").count(),
            4,
            "one strided egress per (batch, weight): {names:?}"
        );
        assert!(!names.iter().any(|n| n == "exl3_f16_to_bf16"));
    }

    #[test]
    fn contract_checks_refuse_bad_geometry() {
        let gpu = MockGpuBackend::new();
        let launch = std::sync::Arc::new(Exl3LaunchState::new(&gpu).unwrap());
        let stage = Exl3DenseStage::new(&gpu, launch, 64, 2560, 4096).unwrap();
        let a = gpu.alloc(8 * 6144 * 2).unwrap();
        let dst = gpu.alloc(8 * 12288 * 2).unwrap();
        // in_dim above the stage's max_in.
        let wide_in = weight(&gpu, 6144, 2560);
        assert!(exl3_dense_linear(&gpu, &wide_in, a, contiguous(dst), 1, &stage, 0).is_err());
        // out_dim above the stage's max_out.
        let wide_out = weight(&gpu, 2560, 12288);
        assert!(exl3_dense_linear(&gpu, &wide_out, a, contiguous(dst), 1, &stage, 0).is_err());
        // Row stride narrower than out_dim.
        let w = weight(&gpu, 2560, 2560);
        let narrow = Exl3DenseOut::strided(dst, 2048);
        assert!(exl3_dense_linear(&gpu, &w, a, narrow, 1, &stage, 0).is_err());
        // Shared-A weights must agree on in_dim.
        let other = weight(&gpu, 1280, 2560);
        let pair = [(w, contiguous(dst)), (other, contiguous(dst))];
        assert!(exl3_dense_linear_shared_a(&gpu, &pair, a, 1, &stage, 0).is_err());
        // m == 0, then the happy path; a refused call must have left no
        // stale section claim behind.
        assert!(exl3_dense_linear(&gpu, &w, a, contiguous(dst), 0, &stage, 0).is_err());
        exl3_dense_linear(&gpu, &w, a, contiguous(dst), 8, &stage, 0).unwrap();
        exl3_dense_linear(&gpu, &w, a, contiguous(dst), 1, &stage, 0).unwrap();
    }

    #[test]
    fn from_exl3_maps_codebook_and_rejects_envelope_breaches() {
        let base = Exl3Weight {
            trellis: DevicePtr(0x1000),
            suh: DevicePtr(0x2000),
            svh: DevicePtr(0x3000),
            in_dim: 2560,
            out_dim: 10240,
            k_bits: 4,
            cb: Exl3Codebook::Mul1,
        };
        assert_eq!(Exl3DenseWeight::from_exl3(&base).unwrap().cb, 2);
        let mcg = Exl3Weight {
            cb: Exl3Codebook::Mcg,
            ..base
        };
        assert_eq!(Exl3DenseWeight::try_from(&mcg).unwrap().cb, 1);
        let inst3 = Exl3Weight {
            cb: Exl3Codebook::Inst3,
            ..base
        };
        assert!(Exl3DenseWeight::from_exl3(&inst3).is_err());
        assert!(Exl3DenseWeight::from_exl3(&Exl3Weight { k_bits: 6, ..base }).is_err());
        let odd = Exl3Weight {
            out_dim: 10200,
            ..base
        };
        assert!(Exl3DenseWeight::from_exl3(&odd).is_err());
    }
}
