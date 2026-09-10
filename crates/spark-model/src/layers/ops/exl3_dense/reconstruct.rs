// SPDX-License-Identifier: AGPL-3.0-only

//! Dense PREFILL tier for the native EXL3 arm: reconstruct the trellis weight
//! to BF16 ONCE per call, then run a fixed-configuration tensor-core BF16 GEMM
//! over every row. Sibling of `exl3_dense.rs` (500-LoC cap).
//!
//! Why: the cooperative `exl3_gemm` decodes every 16x16 trellis tile once per
//! **16-row M slab** (`exl3_gemm_kernel.cuh`), so a 4096-row prefill chunk
//! re-decodes each K=6 dense weight 256 times. Upstream's own policy is
//! reconstruct + `hgemm` above 144 rows (`modules/quant/exl3.py`,
//! `AUTO_RECONSTRUCT_THRESHOLD`), and vllm-exl3 inherits it. Atlas already has
//! the byte-identical reconstruction kernel (`exl3_reconstruct.cu`, GPU-vs-CPU
//! parity in `exl3_reconstruct_parity`) and a fixed-config BF16 GEMM
//! (`dense_gemm_bf16_pipelined`, 128x128 tile, no split-K, no heuristics).
//!
//! Data path per weight (all stream-ordered, no host sync, no allocation):
//!
//! ```text
//!   trellis --exl3_reconstruct_had_k{K}_cb{cb}--> scratch.w_f16  f16 [in, out]
//!           --exl3_f16_to_bf16_t-->                scratch.w_bf16 bf16 [out, in]
//!   A bf16 [m, in] x w_bf16^T --dense_gemm_bf16_pipelined--> C bf16 [m, out]
//!     contiguous dst : C IS the destination (no egress launch at all)
//!     strided dst    : C staged in stage.c_f16 (same 2 B/elem) per row batch
//!                      of stage.rows_cap, then ONE cudaMemcpy2DAsync per batch
//! ```
//!
//! Only the `m > EXL3_GEMV_MAX_M` path changes; the decode arm is untouched.
//!
//! # Numerics (NOT bit-identical to the trellis GEMM)
//!
//! The trellis tier rounds the BF16 activation to f16, rotates it in f16
//! (`A_had = A . H128 . suh`), decodes B in f16, accumulates split-K fp32
//! partials in the kernel's shape-dependent order and rotates the output side
//! (`svh`). This tier folds BOTH Hadamards and sign vectors into the weight
//! (upstream's `reconstruct_had_slice`, f16), rounds that weight ONCE to BF16
//! (2^-9 relative — a rounding the trellis tier never takes), reads the BF16
//! activation unrounded, and accumulates fp32 in the GEMM's fixed K order.
//! Same class as the crate's other gemv-vs-gemm dispatch seams (different
//! rounding points, different reduction order) — greedy output WILL differ
//! from the trellis tier; whether the difference is benign is the GPU A/B's
//! job (HYPOTHESIS: swamped by the 4-bit trellis quantization noise; the
//! agentic gate decides). Run-to-run deterministic by construction: no
//! atomics, no split-K, and each output element's reduction does not depend
//! on `m` (no cuBLASLt heuristics — those pick a kernel per M).
//!
//! `Exl3DenseOut::fp32` is honoured trivially: the GEMM accumulates fp32 and
//! rounds once to BF16, so there is no f16-C range seam to avoid.
//!
//! # Knobs (house convention: env PRESENCE arms; `=0` is not "off")
//!
//! * `ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS=<rows>` — sets the threshold: the
//!   tier takes calls with `m >= rows` (values below
//!   [`EXL3_DENSE_RECONSTRUCT_MIN_ROWS`], including 0 or garbage, clamp to
//!   it: the value never means "off"). UNSET = the default threshold
//!   [`EXL3_DENSE_RECONSTRUCT_DEFAULT_ROWS`] (512), armed by the 2026-09-06
//!   A/B in `.research/exl3_decode_perf/` (`ab_dense_reconstruct/`,
//!   `merged_gate_20260906T004709/`): +7-8% prefill tok/s at 8K/11K on
//!   qwen3.8-flash-next 4.05bpw, decode (m <= 8) untouched, 512 and 1024
//!   byte-identical to each other, model-card agentic gate PASS. The cost is
//!   `2 x max_in x max_out x 2 B` of scratch inside the util pledge (302 MB
//!   at the qwen4_exp maxima 6144 x 12288) and a one-time numerics change
//!   (the weight is rounded to BF16 once; the greedy text of long prompts
//!   differs from the trellis tier's, coherent).
//! * `ATLAS_NO_EXL3_DENSE_RECONSTRUCT` — kill switch (presence): the tier
//!   stays off whatever the threshold says (the trellis GEMM serves every
//!   m > 8 call, the pre-2026-09-06 behaviour).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::exl3::{reconstruct_had_f16_into, transpose_f16_to_bf16_into};

use crate::layers::ops::exl3_matmul::EXL3_GEMV_MAX_M;
use crate::layers::ops::gemm_dense::dense_gemm_bf16_pipelined;
use crate::weight_map::DenseWeight;

use super::stage::Exl3DenseStage;
use super::{Exl3DenseOut, Exl3DenseWeight};

/// Threshold env (presence arms the tier; the value is the minimum `m`).
pub const EXL3_DENSE_RECONSTRUCT_ROWS_ENV: &str = "ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS";
/// Kill switch env (presence disarms the tier regardless of the threshold).
pub const EXL3_DENSE_RECONSTRUCT_KILL_ENV: &str = "ATLAS_NO_EXL3_DENSE_RECONSTRUCT";
/// Smallest row count the tier may take: strictly above the decode arm's
/// GEMV/GEMM tier, which this lever must never touch.
pub const EXL3_DENSE_RECONSTRUCT_MIN_ROWS: usize = EXL3_GEMV_MAX_M + 1;
/// Threshold when the env is unset (measured: see the module doc).
pub const EXL3_DENSE_RECONSTRUCT_DEFAULT_ROWS: usize = 512;
const _: () = assert!(
    EXL3_DENSE_RECONSTRUCT_DEFAULT_ROWS >= EXL3_DENSE_RECONSTRUCT_MIN_ROWS,
    "the default threshold must stay above the decode arm's tier"
);

/// Pure form of the env decision: `rows` is the threshold env's value (if
/// present), `kill` the kill switch's presence. `None` = tier off (kill
/// switch only); unset = [`EXL3_DENSE_RECONSTRUCT_DEFAULT_ROWS`].
pub fn parse_reconstruct_rows(rows: Option<&str>, kill: bool) -> Option<usize> {
    if kill {
        return None;
    }
    let Some(raw) = rows else {
        return Some(EXL3_DENSE_RECONSTRUCT_DEFAULT_ROWS);
    };
    let parsed = raw.trim().parse::<usize>().ok();
    if parsed.is_none() {
        tracing::warn!(
            "{EXL3_DENSE_RECONSTRUCT_ROWS_ENV}={raw:?} is not a row count — the value never \
             means off, so the tier is armed at the minimum of \
             {EXL3_DENSE_RECONSTRUCT_MIN_ROWS} rows ({EXL3_DENSE_RECONSTRUCT_KILL_ENV} disarms)"
        );
    }
    Some(parsed.unwrap_or(0).max(EXL3_DENSE_RECONSTRUCT_MIN_ROWS))
}

/// The process-wide decision, read once.
pub fn reconstruct_rows_from_env() -> Option<usize> {
    static ROWS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        let rows = std::env::var(EXL3_DENSE_RECONSTRUCT_ROWS_ENV).ok();
        let kill = std::env::var_os(EXL3_DENSE_RECONSTRUCT_KILL_ENV).is_some();
        parse_reconstruct_rows(rows.as_deref(), kill)
    })
}

/// Pure dispatch decision: the tier takes a call iff it is armed, `m` reaches
/// the threshold, and `m` is above the decode arm's tier.
pub fn reconstruct_tier_takes(m: usize, threshold: Option<usize>) -> bool {
    m > EXL3_GEMV_MAX_M && threshold.is_some_and(|t| m >= t)
}

/// Scratch bytes for the stage's maxima: `(f16 [in, out], bf16 [out, in])`,
/// each `max_in x max_out x 2`. At qwen4_exp's 6144 x 12288 that is 151 MB
/// each (the actual largest weight, q_proj 2560 x 12288, needs 63 MB — the
/// stage API carries only the axis maxima, so this over-provisions by the
/// in/out cross term; tightening it needs a max-weight-elements hint from
/// the loader).
pub fn reconstruct_scratch_bytes(max_in: usize, max_out: usize) -> (usize, usize) {
    let each = max_in * max_out * 2;
    (each, each)
}

/// Stage-owned scratch for the tier — allocated ONCE at stage construction
/// (inside the util pledge, before the KV budget), only when armed.
#[derive(Debug)]
pub struct Exl3ReconScratch {
    /// f16 `[in, out]` reconstruction target (`elems` elements).
    pub w_f16: DevicePtr,
    /// bf16 `[out, in]` GEMM operand (`elems` elements).
    pub w_bf16: DevicePtr,
    /// Capacity of each slab in elements (`max_in * max_out`).
    pub elems: usize,
    /// Minimum `m` the tier takes (>= [`EXL3_DENSE_RECONSTRUCT_MIN_ROWS`]).
    pub threshold: usize,
}

impl Exl3ReconScratch {
    /// Allocate both slabs (all-or-nothing) and resolve the GEMM kernel.
    pub fn new(
        gpu: &dyn GpuBackend,
        max_in: usize,
        max_out: usize,
        threshold: usize,
    ) -> Result<Self> {
        ensure!(
            threshold >= EXL3_DENSE_RECONSTRUCT_MIN_ROWS,
            "EXL3 dense reconstruct tier: threshold {threshold} would reach the decode arm \
             (minimum {EXL3_DENSE_RECONSTRUCT_MIN_ROWS})"
        );
        // Load-time probe: a target without the GEMM refuses here, not on
        // the first long prompt (the run resolves it by name per call, the
        // module-wide convention — a map lookup, and it keeps the launch
        // plan visible to the mock-backend tests).
        gemm_kernel(gpu).map_err(|e| {
            anyhow::anyhow!(
                "EXL3 dense reconstruct tier needs gemm::dense_gemm_bf16_pipelined on this \
                 target — unset {EXL3_DENSE_RECONSTRUCT_ROWS_ENV}: {e}"
            )
        })?;
        let (f16_bytes, bf16_bytes) = reconstruct_scratch_bytes(max_in, max_out);
        let w_f16 = gpu.alloc(f16_bytes)?;
        let w_bf16 = match gpu.alloc(bf16_bytes) {
            Ok(p) => p,
            Err(e) => {
                gpu.free(w_f16).ok();
                return Err(e);
            }
        };
        tracing::info!(
            "EXL3 dense reconstruct tier ARMED: calls with m >= {threshold} rows reconstruct \
             the trellis weight to BF16 once and run dense_gemm_bf16_pipelined; scratch \
             {:.1} MB (f16 [in, out] + bf16 [out, in] at {max_in} x {max_out}). Numerics \
             differ from the trellis GEMM (see ops/exl3_dense/reconstruct.rs); \
             {EXL3_DENSE_RECONSTRUCT_KILL_ENV} disarms",
            (f16_bytes + bf16_bytes) as f64 / 1e6,
        );
        Ok(Self {
            w_f16,
            w_bf16,
            elems: max_in * max_out,
            threshold,
        })
    }

    /// Does this call take the tier?
    pub fn takes(&self, m: usize) -> bool {
        reconstruct_tier_takes(m, Some(self.threshold))
    }

    pub fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.w_f16)?;
        gpu.free(self.w_bf16)
    }
}

/// The fixed-configuration tensor-core BF16 GEMM (128x128 tile, no split-K).
/// Resolved per launch — a map lookup, the converter wrappers' convention,
/// so the recorded plan is the launch plan.
fn gemm_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    gpu.kernel("gemm", "dense_gemm_bf16_pipelined")
}

/// The tier body for one shared-A weight group. Caller holds the stage's
/// dispatch section (it protects the stage-shared scratch — `w_f16`,
/// `w_bf16`, `c_f16` — against a second host thread, even though nothing
/// here is a cooperative launch) and has validated the group geometry.
pub(super) fn run_reconstruct_tier(
    gpu: &dyn GpuBackend,
    ws: &[(Exl3DenseWeight, Exl3DenseOut)],
    a_bf16: DevicePtr,
    m: usize,
    stage: &Exl3DenseStage,
    rs: &Exl3ReconScratch,
    stream: u64,
) -> Result<()> {
    for (w, out) in ws {
        let (k, n) = (w.in_dim, w.out_dim);
        ensure!(
            k * n <= rs.elems,
            "EXL3 dense reconstruct tier: [{k} -> {n}] exceeds the scratch capacity of {} \
             elements — size the stage from model-wide maxima",
            rs.elems
        );
        reconstruct_had_f16_into(
            gpu, w.trellis, w.suh, w.svh, k, n, w.k_bits, w.cb, rs.w_f16, stream,
        )?;
        transpose_f16_to_bf16_into(gpu, rs.w_f16, rs.w_bf16, k, n, stream)?;
        let weight = DenseWeight { weight: rs.w_bf16 };
        let ld = out.ld.unwrap_or(n);
        if ld == n {
            // Contiguous: the GEMM's BF16 C is the destination itself.
            dense_gemm_bf16_pipelined(
                gpu,
                gemm_kernel(gpu)?,
                a_bf16,
                &weight,
                out.ptr,
                m as u32,
                n as u32,
                k as u32,
                stream,
            )?;
            continue;
        }
        // Pitched: stage BF16 C in `c_f16` (2 B/elem, `rows_cap x max_out`)
        // per row batch, then one 2-D copy into the arena's column block.
        let mut r0 = 0usize;
        while r0 < m {
            let rows = (m - r0).min(stage.rows_cap);
            dense_gemm_bf16_pipelined(
                gpu,
                gemm_kernel(gpu)?,
                a_bf16.offset(r0 * k * 2),
                &weight,
                stage.c_f16,
                rows as u32,
                n as u32,
                k as u32,
                stream,
            )?;
            gpu.copy_d2d_2d_async(
                stage.c_f16,
                n * 2,
                out.ptr.offset(r0 * ld * 2),
                ld * 2,
                n * 2,
                rows,
                stream,
            )?;
            r0 += rows;
        }
    }
    Ok(())
}
