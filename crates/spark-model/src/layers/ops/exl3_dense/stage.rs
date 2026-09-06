// SPDX-License-Identifier: AGPL-3.0-only

//! Staging slabs for the native dense-linear arm: the EXL3 kernels read RAW
//! fp16 A and write fp16/fp32 C, while every Atlas consumer is BF16, so each
//! projection needs a bf16->f16 ingress slab and (for the strided / f32
//! paths) a C staging slab. Allocated ONCE at load inside the util pledge
//! (before the KV budget), sized from the prefill chunk x the largest
//! projection dims; rows above the capacity are served in row batches.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::exl3_matmul::EXL3_GEMV_MAX_M;

use super::launch_state::Exl3LaunchState;
use super::reconstruct::{Exl3ReconScratch, reconstruct_rows_from_env};

/// Default row capacity of the staging slabs (`ATLAS_EXL3_DENSE_STAGE_ROWS`
/// overrides). Row batching above it costs one extra launch triple per
/// batch, so this is a memory/launch-count knob, not a correctness one.
pub const EXL3_DENSE_STAGE_ROWS_DEFAULT: usize = 4096;

/// Model-shared dense staging: ONE per model (every GDN/attention layer
/// dispatches through it under the launch state's section, so one slab set
/// suffices). Nothing on the hot path may alloc or sync (901 playbook).
#[derive(Debug)]
pub struct Exl3DenseStage {
    /// Shared locks / fence / section mutex (also used by the MoE arm).
    pub launch: std::sync::Arc<Exl3LaunchState>,
    /// fp16 RAW activation ingress `[rows_cap, max_in]`.
    pub a_f16: DevicePtr,
    /// fp16 `A_had` rotation scratch `[rows_cap, max_in]`, SEPARATE from
    /// `a_f16` so a shared-A group (GDN qkv + z; attention q/k/v) can rotate
    /// the same raw A under each weight's own `suh` without re-ingress.
    pub a_had_f16: DevicePtr,
    /// fp16 C staging `[rows_cap, max_out]` for the STRIDED destination
    /// path (a contiguous destination takes f16 C directly + in-place
    /// convert, the lm_head precedent).
    pub c_f16: DevicePtr,
    /// fp32 C `[EXL3_GEMV_MAX_M, max_out]` for the small-m GEMV tier (and
    /// its GEMM fallthrough) — fp32 output for the decode-critical rows.
    /// Also the GEMM tier's C for `Exl3DenseOut::fp32` destinations (the
    /// residual-bound out_proj / o_proj, which upstream pins to fp32 out):
    /// `c_f32_elems` says how many f32 elements it holds.
    pub c_f32: DevicePtr,
    /// Capacity of `c_f32` in elements: `max(8 * max_out, rows_cap *
    /// max_out_f32)`.
    pub c_f32_elems: usize,
    /// Row capacity of `a_f16` / `a_had_f16` / `c_f16`.
    pub rows_cap: usize,
    /// Largest `in_dim` any weight served through this stage may have.
    pub max_in: usize,
    /// Largest `out_dim` any weight served through this stage may have.
    pub max_out: usize,
    /// Reconstruct-to-BF16 prefill tier scratch (`ops/exl3_dense/reconstruct.rs`)
    /// — `Some` only when `ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS` armed it at
    /// construction; `None` = the trellis GEMM serves every m > 8 (default).
    pub recon: Option<Exl3ReconScratch>,
}

impl Exl3DenseStage {
    /// Allocate the slabs, all-or-nothing with rollback (the `Exl3LmHead::new`
    /// pattern). `rows_cap` is the loader's prefill-chunk hint; the env
    /// override wins when set. One named call site so the alloc ledger shows
    /// one legible row per slab.
    pub fn new(
        gpu: &dyn GpuBackend,
        launch: std::sync::Arc<Exl3LaunchState>,
        rows_cap: usize,
        max_in: usize,
        max_out: usize,
    ) -> Result<Self> {
        Self::new_with_fp32(gpu, launch, rows_cap, max_in, max_out, 0)
    }

    /// [`Self::new`] plus `max_out_f32`: the widest `out_dim` any weight may
    /// project through the GEMM tier with an fp32 C (`Exl3DenseOut::fp32`,
    /// the residual-bound out_proj / o_proj). 0 keeps the f32 slab at the
    /// GEMV tier's 8 rows and makes fp32 GEMM destinations refuse.
    pub fn new_with_fp32(
        gpu: &dyn GpuBackend,
        launch: std::sync::Arc<Exl3LaunchState>,
        rows_cap: usize,
        max_in: usize,
        max_out: usize,
        max_out_f32: usize,
    ) -> Result<Self> {
        Self::new_with_reconstruct(
            gpu,
            launch,
            rows_cap,
            max_in,
            max_out,
            max_out_f32,
            reconstruct_rows_from_env(),
        )
    }

    /// [`Self::new_with_fp32`] with the reconstruct tier's threshold passed
    /// explicitly instead of read from the environment (`None` = tier off):
    /// the primitive every other constructor lowers to, and the one tests
    /// call so they never touch process env.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_reconstruct(
        gpu: &dyn GpuBackend,
        launch: std::sync::Arc<Exl3LaunchState>,
        rows_cap: usize,
        max_in: usize,
        max_out: usize,
        max_out_f32: usize,
        reconstruct_rows: Option<usize>,
    ) -> Result<Self> {
        let rows_cap = std::env::var("ATLAS_EXL3_DENSE_STAGE_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(rows_cap);
        ensure!(
            rows_cap >= EXL3_GEMV_MAX_M,
            "EXL3 dense stage: rows_cap {rows_cap} < the GEMV tier's {EXL3_GEMV_MAX_M} rows"
        );
        ensure!(
            max_in >= 128
                && max_in.is_multiple_of(128)
                && max_out >= 128
                && max_out.is_multiple_of(128),
            "EXL3 dense stage: max_in {max_in} / max_out {max_out} must be multiples of 128"
        );
        let mut owned: Vec<DevicePtr> = Vec::new();
        let mut alloc = |bytes: usize| -> Result<DevicePtr> {
            match gpu.alloc(bytes) {
                Ok(p) => {
                    owned.push(p);
                    Ok(p)
                }
                Err(e) => {
                    for p in owned.drain(..) {
                        gpu.free(p).ok();
                    }
                    Err(e)
                }
            }
        };
        let a_f16 = alloc(rows_cap * max_in * 2)?;
        let a_had_f16 = alloc(rows_cap * max_in * 2)?;
        let c_f16 = alloc(rows_cap * max_out * 2)?;
        let c_f32_elems = (EXL3_GEMV_MAX_M * max_out).max(rows_cap * max_out_f32);
        let c_f32 = alloc(c_f32_elems * 4)?;
        let recon = match reconstruct_rows {
            Some(threshold) => match Exl3ReconScratch::new(gpu, max_in, max_out, threshold) {
                Ok(r) => Some(r),
                Err(e) => {
                    for p in [a_f16, a_had_f16, c_f16, c_f32] {
                        gpu.free(p).ok();
                    }
                    return Err(e);
                }
            },
            None => {
                tracing::info!(
                    "EXL3 dense reconstruct tier off (default): every m > 8 call runs the \
                     cooperative trellis GEMM; ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS=<rows> arms it"
                );
                None
            }
        };
        let total = rows_cap * (max_in * 4 + max_out * 2) + c_f32_elems * 4;
        tracing::info!(
            "EXL3 native dense stage allocated: {rows_cap} rows x (in {max_in}, out \
             {max_out}; fp32-C out {max_out_f32}) = {:.1} MB slabs (A + A_had + f16 C + \
             f32 C), shared across all GDN/attention layers; rows above {rows_cap} are \
             batched",
            total as f64 / 1e6,
        );
        Ok(Self {
            launch,
            a_f16,
            a_had_f16,
            c_f16,
            c_f32,
            c_f32_elems,
            rows_cap,
            max_in,
            max_out,
            recon,
        })
    }

    /// Get the model-shared stage, creating it on first use (the loader
    /// threads one `Option` cache through its per-layer loop). The geometry
    /// must not shrink between calls — pass the model-wide maxima.
    pub fn get_or_create(
        cache: &mut Option<std::sync::Arc<Exl3DenseStage>>,
        gpu: &dyn GpuBackend,
        launch: &std::sync::Arc<Exl3LaunchState>,
        rows_cap: usize,
        max_in: usize,
        max_out: usize,
        max_out_f32: usize,
    ) -> Result<std::sync::Arc<Exl3DenseStage>> {
        if let Some(s) = cache {
            ensure!(
                s.max_in >= max_in
                    && s.max_out >= max_out
                    && s.c_f32_elems >= s.rows_cap * max_out_f32,
                "EXL3 dense stage: geometry grew between layers (stage in {} out {} vs \
                 requested in {max_in} out {max_out} fp32-out {max_out_f32}) — size the \
                 stage from model-wide maxima",
                s.max_in,
                s.max_out,
            );
            ensure!(
                std::sync::Arc::ptr_eq(&s.launch, launch),
                "EXL3 dense stage: a second launch state was passed for one model"
            );
            return Ok(s.clone());
        }
        let s = std::sync::Arc::new(Self::new_with_fp32(
            gpu,
            launch.clone(),
            rows_cap,
            max_in,
            max_out,
            max_out_f32,
        )?);
        *cache = Some(s.clone());
        Ok(s)
    }

    /// Free the slabs (NOT the shared launch state — release that once every
    /// holder is gone). Without an explicit caller this is reclaimed by
    /// `sweep_unreleased` at teardown (documented backstop).
    pub fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [self.a_f16, self.a_had_f16, self.c_f16, self.c_f32] {
            gpu.free(p)?;
        }
        if let Some(r) = &self.recon {
            r.release(gpu)?;
        }
        Ok(())
    }
}
