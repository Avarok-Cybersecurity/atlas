// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: serve a native-FP8 dense FFN over 5 to 32 DECODE or VERIFY
//! rows on the sm_89 tensor-core GEMM
//! (`kernels/ada/common/fp8_gemm_ada_m16.cu`: the 16-row tile for 5 to 16
//! rows, the 32-row tile for 17 to 32), behind the `ffn_fp8_m16_tc` row (off by
//! default on every target).
//!
//! # What it replaces, and why
//!
//! A decode step over `n` streams and a speculative verify over `n` rows run
//! each layer's dense FFN over `n` rows. On an FP8 layer those rows reach:
//!
//! | rows | kernel today | cost a projection (RTX 4060 Ti 16 GB, 4B gate/up) |
//! |---|---|---|
//! | 2..4 | `w8a16_gemv_batch4` | 92 us, the weight-bandwidth floor (91 us at 260 GB/s) |
//! | 5..8 | the W8A8 arm, or the batch-M GEMV with its rows on | 143 us; 97 to 184 us |
//! | 9..64 | the W8A8 arm | 143 us at every width (162 us on down) |
//!
//! The W8A8 arm (`fp8_gemm_ada_w8a8`) runs a 64-row padded tensor-core tile,
//! so it does four times the MMA work a 16-row tile needs and is not bandwidth
//! bound; it also quantizes the activations to E4M3. The batch-M GEMV
//! (`fp8_gemv_ada_batchm`) is exact but spends FP32 FMAs per row per weight
//! byte and is compute bound above four rows.
//!
//! `fp8_gemm_ada_m16` keeps the activations BF16, widens the E4M3 weights in
//! registers and runs one m16n8k16 MMA per 128 weight bytes, reading each
//! weight once for up to 16 rows. Its `.cu` header has the geometry and the
//! measurement behind the operand choice. The 32-row entries
//! (`fp8_gemm_ada_m32`, `_m32_dual`) put a second A tile on every dequantized
//! weight fragment, so 17 to 32 rows (a 3-draft verify at 8 streams, a DFlash
//! verify) also read each weight once; rows 0 to 15 are bit-equal to the
//! 16-row kernel's.
//!
//! # Scope
//!
//! Decode steps and verify passes only. Four call sites ask the arm, none of
//! them on a prompt pass:
//!
//! * `DenseFfnLayer::fp8_verify_arm` (`w4.rs`), the hook `forward_k2`,
//!   `forward_k3` and `forward_km` try first: 2 to 8 rows from either layer
//!   kind, of which this arm takes 5 to 8 (the wide arms below reach 32);
//! * the wide arms that go straight to `forward_prefill`, through
//!   [`FfnComponent::try_forward_fp8_m16_tc`]: the SSM multi-sequence decode
//!   (`qwen3_ssm/trait_decode_multi_seq.rs`), the SSM batched verify
//!   (`qwen3_ssm/trait_decode_batched.rs`) and the attention layers' batched
//!   decode and verify (`qwen3_attention/trait_impl/multi_seq/ffn.rs`).
//!
//! Rows 2..4 are left to `w8a16_gemv_batch4`: it is bit-identical to the
//! single-stream decode GEMV and already at the floor, so there is nothing to
//! buy there and an exactness to lose (`AVAROK_FFN_FP8_M16_TC_MIN_ROWS=2`
//! lowers the bound for a measurement). Above 32 rows the arm declines and the
//! caller keeps its path.
//!
//! With the row on, this arm is asked BEFORE `ffn_fp8_verify_gemv` and
//! `ffn_fp8_decode_gemv` at the widths it takes; those rows keep 2..4.
//!
//! # Numerics
//!
//! Not bit-identical to the decode GEMV: an MMA reduces 16 K-products in the
//! tensor core's order, and the block scale is folded once per 128-wide block
//! on an FP32 accumulator instead of multiplied into every weight. The
//! contract is the M16 tensor-core one ([`super::m16_tc::oracle`]): at most 2
//! BF16 ULP from `fp8_gemv_ada` per element, or under the FP32 accumulation
//! floor for a cancelled output. `examples/fp8_ada_m16_tc_microtest.rs`
//! measures it at every M in 1..=32 against the GEMV and a host FP64
//! reference. A row's bits do not depend on how many rows share the launch,
//! so a verify row equals the same token decoded in a batch on this route.
//!
//! The SwiGLU product between the projections is the layer's own activation
//! multiply (`act_mul`), on the BF16 gate and up rows, exactly as the decode
//! forward's default arm and the prefill ladder compute it.
//!
//! [`FfnComponent::try_forward_fp8_m16_tc`]: crate::layers::FfnComponent::try_forward_fp8_m16_tc

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::DenseFfnLayer;
use super::ada_kernels::{Entry, cached_target_kernel};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// Module (file stem) of the kernel.
pub const MODULE: &str = "fp8_gemm_ada_m16";
/// One projection: `(A, B, block_scale, C, M, N, K)`.
pub const PLAIN: Entry = (MODULE, "fp8_gemm_ada_m16");
/// Gate and up of the same rows in one launch.
pub const DUAL: Entry = (MODULE, "fp8_gemm_ada_m16_dual");
/// One projection over 17..=32 rows: `(A, B, block_scale, C, M, N, K)`.
pub const PLAIN32: Entry = (MODULE, "fp8_gemm_ada_m32");
/// Gate and up of the same 17..=32 rows in one launch.
pub const DUAL32: Entry = (MODULE, "fp8_gemm_ada_m32_dual");
/// Most rows the 16-row entries serve: their M tile.
pub const TILE_ROWS: usize = 16;
/// Most rows one launch serves: the 32-row entries' M tile.
pub const MAX_ROWS: usize = 32;
/// Output columns one CTA covers (4 warps, one n8 MMA tile each).
pub const N_TILE: u32 = 32;
/// Threads in a CTA.
pub const THREADS: u32 = 128;
/// The kernel folds one block scale per 128 K, so K must be whole blocks.
pub const K_BLOCK: usize = 128;

/// The four handles the arm launches: a pair for 5..=16 rows and a pair for
/// 17..=32.
#[derive(Clone, Copy, Debug)]
pub struct TcKernels {
    /// `fp8_gemm_ada_m16`.
    pub plain: KernelHandle,
    /// `fp8_gemm_ada_m16_dual`.
    pub dual: KernelHandle,
    /// `fp8_gemm_ada_m32`.
    pub plain32: KernelHandle,
    /// `fp8_gemm_ada_m32_dual`.
    pub dual32: KernelHandle,
}

impl TcKernels {
    /// The handles on this backend, through its op cache: a map read once
    /// [`prime`] ran. `None` where the target did not compile the module (no
    /// lookup is issued then).
    pub fn cached(gpu: &dyn GpuBackend) -> Option<Self> {
        if !gpu.has_module(MODULE) {
            return None;
        }
        Some(Self {
            plain: cached_target_kernel(gpu, PLAIN),
            dual: cached_target_kernel(gpu, DUAL),
            plain32: cached_target_kernel(gpu, PLAIN32),
            dual32: cached_target_kernel(gpu, DUAL32),
        })
    }

    /// The (plain, dual) pair that serves `rows`: the 16-row tile up to
    /// [`TILE_ROWS`], the 32-row tile above.
    pub fn pair_for(&self, rows: usize) -> (KernelHandle, KernelHandle) {
        if rows <= TILE_ROWS {
            (self.plain, self.dual)
        } else {
            (self.plain32, self.dual32)
        }
    }
}

/// Construction-time lookup of all four handles, so the forward never issues a
/// first lookup. Called only with the row on (`ada_kernels::Fp8FfnKernels::resolve`).
pub(crate) fn prime(gpu: &dyn GpuBackend) {
    let _ = TcKernels::cached(gpu);
}

/// The admission rule's row-count half, as a pure function so a CPU test can
/// pin every clause without a backend:
///
/// * `row_on`: the `ffn_fp8_m16_tc` row;
/// * `native_fp8` and not `packed_q2`: the layer holds the block-scaled FP8
///   overlay and no keep-packed Q2_0 weights (those have their own arm);
/// * `rows` in `min_rows..=MAX_ROWS` (32), and never fewer than 2: one row is
///   the M=1 decode forward.
pub(crate) fn admits(
    row_on: bool,
    native_fp8: bool,
    packed_q2: bool,
    rows: usize,
    min_rows: usize,
) -> bool {
    row_on && native_fp8 && !packed_q2 && (min_rows.max(2)..=MAX_ROWS).contains(&rows)
}

/// A block-scaled `[n, k]` weight at the shape the FFN reads.
fn weight_ok(w: &Fp8Weight, n: usize, k: usize) -> bool {
    w.scale_format == WeightQuantFormat::Fp8BlockScaled && w.n as usize == n && w.k as usize == k
}

impl DenseFfnLayer {
    /// The FFN over `rows` decode or verify rows on the 16-row (up to 16) or
    /// 32-row (17 to 32) tensor-core GEMM, into `moe_output` (where every small-batch entry point writes),
    /// or `Ok(false)` with nothing launched: row off, not a native-FP8 layer,
    /// a row count outside the arm's range, or the kernels absent.
    ///
    /// The structural clauses are tested before the row table is read, so a
    /// layer that can never take the arm never touches it.
    pub(crate) fn fp8_m16_tc_arm(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let structural = admits(
            true,
            self.fp8_weights.is_some(),
            self.q2_weights.is_some(),
            rows,
            2,
        );
        if !structural || !ops::port_rows::ffn_fp8_m16_tc::row() {
            return Ok(false);
        }
        let Some(k) = TcKernels::cached(ctx.gpu) else {
            return Ok(false);
        };
        let min_rows = ops::port_rows::ffn_fp8_m16_tc::min_rows();
        self.fp8_m16_tc_arm_with(true, min_rows, &k, input, rows, ctx, stream)
    }

    /// [`Self::fp8_m16_tc_arm`] over an explicit row value, lower bound and
    /// handles, so a CPU test drives every polarity without sealing the
    /// process-wide row table.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fp8_m16_tc_arm_with(
        &self,
        row_on: bool,
        min_rows: usize,
        k: &TcKernels,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !admits(
            row_on,
            self.fp8_weights.is_some(),
            self.q2_weights.is_some(),
            rows,
            min_rows,
        ) {
            return Ok(false);
        }
        let Some(fp8w) = self.fp8_weights.as_ref() else {
            return Ok(false);
        };
        let (hidden, inter) = (ctx.config.hidden_size, ctx.config.intermediate_size);
        let (g, u, d) = (&fp8w.gate_proj, &fp8w.up_proj, &fp8w.down_proj);
        // Both handles, the layer's activation multiply, no adapter (one is
        // never installed on an FP8 layer, but a decline is the safe answer),
        // whole scale blocks along both reductions, and the three weights at
        // the shapes the FFN reads. Anything else declines rather than launch
        // and be wrong.
        let (plain, dual) = k.pair_for(rows);
        let ready = plain.0 != 0
            && dual.0 != 0
            && self.act_mul.0 != 0
            && self.lora.is_none()
            && hidden.is_multiple_of(K_BLOCK)
            && inter.is_multiple_of(K_BLOCK)
            && weight_ok(g, inter, hidden)
            && weight_ok(u, inter, hidden)
            && weight_ok(d, hidden, inter);
        if !ready {
            return Ok(false);
        }
        let (h, i, m) = (hidden as u32, inter as u32, rows as u32);
        let gpu = ctx.gpu;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        gemm_m16_dual(gpu, dual, input, g, gate_out, u, up_out, m, i, h, stream)?;
        ops::silu_mul(gpu, self.act_mul, gate_out, up_out, gate_out, m * i, stream)?;
        gemm_m16(
            gpu,
            plain,
            gate_out,
            d,
            ctx.buffers.moe_output(),
            m,
            h,
            i,
            stream,
        )?;
        if ctx.stats.once("log:ffn_fp8_m16_tc") {
            tracing::info!(
                "[avarok] dense FFN decode/verify rows: FP8 16/32-row tensor-core GEMM ({MODULE}, \
                 {min_rows}..{TILE_ROWS} rows on the 16-row tile, {}..{MAX_ROWS} on the 32-row tile, \
                 gate+up in one launch, then down); one weight pass, \
                 BF16 activations, REASSOCIATED against the decode GEMV (at most 2 BF16 ULP). \
                 AVAROK_FFN_FP8_M16_TC=0 restores the prefill ladder (the W8A8 arm at these widths).",
                TILE_ROWS + 1
            );
        }
        Ok(true)
    }
}

impl crate::layers::FfnComponent {
    /// The decode or verify FFN over `rows` rows on the FP8 16-row
    /// tensor-core GEMM ([`DenseFfnLayer::fp8_m16_tc_arm`]), for a caller
    /// that would otherwise go straight to `forward_prefill`. `Ok(false)` =
    /// nothing launched and the caller keeps its path: always, for an MoE or
    /// absent FFN. Never call it on a prompt pass.
    pub fn try_forward_fp8_m16_tc(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        match self {
            Self::Dense(d) => d.fp8_m16_tc_arm(input, rows, ctx, stream),
            _ => Ok(false),
        }
    }
}

/// `out[m, n] = a[m, k] x E4M3(w[n, k])^T * scales`: `m` in 1..=16 with
/// [`PLAIN`], 1..=32 with [`PLAIN32`] (same arguments and grid).
///
/// Kernel: `fp8_gemm_ada_m16(A, B, block_scale, C, M, N, K)`
/// Grid: (ceil(N/32), 1, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gemm_m16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    w: &Fp8Weight,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, N_TILE), 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.weight)
        .arg_ptr(w.row_scale)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Gate and up of the same `[m, k]` rows in one launch ([`DUAL`] up to 16
/// rows, [`DUAL32`] up to 32).
///
/// Kernel: `fp8_gemm_ada_m16_dual(A, B1, B1_scale, C1, B2, B2_scale, C2, M, N, K)`
/// Grid: (ceil(N/32), 1, 2)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gemm_m16_dual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    w1: &Fp8Weight,
    out1: DevicePtr,
    w2: &Fp8Weight,
    out2: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, N_TILE), 1, 2])
        .block([THREADS, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w1.weight)
        .arg_ptr(w1.row_scale)
        .arg_ptr(out1)
        .arg_ptr(w2.weight)
        .arg_ptr(w2.row_scale)
        .arg_ptr(out2)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[cfg(test)]
#[path = "fp8_m16_tc_tests.rs"]
mod tests;
