// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the dense FFN's packed-weight arm. When a loader route
//! installs gate/up/down as [`W4Linear`] weights (`DenseFfnLayer::set_w4`),
//! every entry point of the layer serves them here: M = 1 decode, the K = 2,
//! 3 and m-row verify and batched decode, and prefill. A child module of
//! `dense_ffn.rs` (allow-listed, hook lines only) so it can read the layer's
//! private handles.
//!
//! # Precedence (the same at every entry point)
//!
//! packed Q2 > native FP8 > **W4** > native BF16 overlay > NVFP4. `set_w4`
//! detaches a BF16 overlay, so the last two never coexist with W4; Q2 and FP8
//! routes never install W4 (the loader's FP8-wins rule). Using one predicate,
//! [`DenseFfnLayer::w4_active`], at every hook keeps decode and prefill from
//! ever taking different formats on one layer.
//!
//! # Routing by rows ([`select`])
//!
//! * gate+up: `decode_pair` when `rows` fits both decode windows, else each
//!   projection on its own.
//! * each projection: `decode` for `rows <= max_decode_rows()`, else
//!   `prefill` in even chunks of at most `select::prefill_chunk_cap` rows; a
//!   declined chunk (`Ok(false)`) dequantizes that projection ONCE into the
//!   shared BF16 scratch and runs the layer's existing BF16 GEMM ladder
//!   (cuBLASLt when the FFN dispatch says so, else the tensor-core kernel,
//!   else scalar) for that chunk.
//! * SwiGLU between them is the layer's `act_mul` over `rows * inter`, staged
//!   once (no fused silu-input kernel), at every row count.
//!
//! Row invariance follows: a verify row (rows <= the window) runs exactly the
//! kernels the M = 1 decode runs, with its own row slice, so speculative
//! decoding at temperature 0 matches plain decoding on this route.
//!
//! # The FP8 verify GEMV (master plan 3.7.3)
//!
//! [`DenseFfnLayer::fp8_verify_arm`] offers a native-FP8 layer's 2..8-row
//! verify to `ada_kernels::verify::forward_km_fp8` (lane L15). That function
//! returns `Ok(false)` without launching anything when the
//! `ffn_fp8_verify_gemv` row is off, so the row-off path is the redirect to
//! `forward_prefill` exactly as before.

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::port::w4::scratch::{self, ScratchSpec};
use crate::port::w4::select::{self, PairCall, ProjCall};
use crate::port::w4::{self, W4Box, W4Linear, W4Scratch};
use crate::weight_map::DenseWeight;

/// The installed packed gate/up/down plus the scratch view reserved for them.
pub struct W4Ffn {
    gate: W4Box,
    up: W4Box,
    down: W4Box,
    spec: ScratchSpec,
    /// Set by `finalize_w4_load` (or the first eager forward); read by every
    /// forward, so a captured decode needs no host-side cache lookup.
    scratch: OnceLock<W4Scratch>,
}

impl W4Ffn {
    fn new(gate: W4Box, up: W4Box, down: W4Box) -> Self {
        let spec = ScratchSpec::for_all(&[&*gate, &*up, &*down]);
        Self {
            gate,
            up,
            down,
            spec,
            scratch: OnceLock::new(),
        }
    }

    pub fn gate(&self) -> &dyn W4Linear {
        &*self.gate
    }

    pub fn up(&self) -> &dyn W4Linear {
        &*self.up
    }

    pub fn down(&self) -> &dyn W4Linear {
        &*self.down
    }

    /// Resident bytes of the three packed weights.
    pub fn device_bytes(&self) -> usize {
        self.gate.device_bytes() + self.up.device_bytes() + self.down.device_bytes()
    }

    /// What this layer's calls need from the shared scratch.
    pub fn scratch_spec(&self) -> ScratchSpec {
        self.spec
    }

    /// The reserved scratch view. Reserving needs an allocation outside graph
    /// capture; a capture that reaches an unreserved layer is refused with
    /// the fix in the message instead of failing inside the driver.
    fn scratch(&self, gpu: &dyn GpuBackend, capturing: bool, stream: u64) -> Result<W4Scratch> {
        if let Some(s) = self.scratch.get() {
            return Ok(*s);
        }
        anyhow::ensure!(
            !capturing,
            "dense FFN W4 ({}): scratch not reserved before CUDA graph capture; \
             the loader must call DenseFfnLayer::finalize_w4_load (or \
             finalize_q4k_load) after set_w4",
            self.gate.format()
        );
        let s = scratch::acquire(gpu, &self.spec, stream)?;
        let _ = self.scratch.set(s);
        Ok(*self.scratch.get().expect("set above"))
    }
}

impl DenseFfnLayer {
    /// Install packed gate (`[inter, hidden]`), up (`[inter, hidden]`) and
    /// down (`[hidden, inter]`) weights. From here every entry point of the
    /// layer serves them (module docs). A native BF16 overlay, if one was
    /// installed, is detached: the return value is its size in bytes, which
    /// the caller (who owns those buffers: store aliases or a D2D snapshot)
    /// may now free. 0 when there was none.
    ///
    /// Call it BEFORE `finalize_q4k_load` / `finalize_nvfp4_mmq_load`: those
    /// then reserve the W4 scratch and skip the NVFP4 repacks (the loader
    /// builds the NVFP4 fallback NULL on a packed route, so a repack would
    /// read null pointers). The three weights' shapes are checked against the
    /// model sizes at the first forward; inconsistent shapes are logged here.
    pub fn set_w4(&mut self, gate: W4Box, up: W4Box, down: W4Box) -> usize {
        if !select::ffn_consistent(&*gate, &*up, &*down) {
            tracing::error!(
                "dense FFN W4: inconsistent shapes gate [{}, {}], up [{}, {}], down [{}, {}]; \
                 the first forward will refuse this layer",
                gate.n(),
                gate.k(),
                up.n(),
                up.k(),
                down.n(),
                down.k()
            );
        }
        let freed = match self.bf16_weights.take() {
            Some(_) => (gate.n() * gate.k() + up.n() * up.k() + down.n() * down.k()) * 2,
            None => 0,
        };
        self.w4 = Some(W4Ffn::new(gate, up, down));
        freed
    }

    /// Whether packed W4 weights are installed.
    pub fn has_w4(&self) -> bool {
        self.w4.is_some()
    }

    /// The installed packed weights (residency, self-check, metrics).
    pub fn w4(&self) -> Option<&W4Ffn> {
        self.w4.as_ref()
    }

    /// Reserve the shared W4 scratch for this layer at load, before the KV
    /// cache is sized and before any graph capture. Idempotent. Returns the
    /// bytes reserved for this layer's spec (an upper bound on what it adds:
    /// layers with the same shapes share one set of regions); 0 without W4.
    pub fn finalize_w4_load(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<usize> {
        let Some(w) = self.w4.as_ref() else {
            return Ok(0);
        };
        w.scratch(gpu, false, stream)?;
        Ok(w.spec.device_bytes())
    }

    /// The W4 arm's predicate, used by every entry point (module docs).
    pub(super) fn w4_active(&self) -> bool {
        self.w4.is_some() && self.q2_weights.is_none() && self.fp8_weights.is_none()
    }

    /// The native-FP8 batch-M verify GEMV for `m` rows (3.7.3), or `false`
    /// to keep the current path. Launches nothing when the row is off, the
    /// layer is not native FP8, or the kernels are absent.
    pub(super) fn fp8_verify_arm(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if self.fp8_weights.is_none() || self.q2_weights.is_some() {
            return Ok(false);
        }
        super::ada_kernels::verify::forward_km_fp8(
            self,
            input,
            m as usize,
            ctx.buffers.moe_output(),
            ctx,
            stream,
        )
    }

    /// M = 1 decode through the W4 weights; returns the output buffer the
    /// way `forward` does.
    pub(super) fn forward_w4_decode(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.forward_w4_rows(input, 1, ctx, stream)?;
        Ok(ctx.buffers.moe_output())
    }

    /// `rows` rows through the W4 weights: `input` `[rows, hidden]`, output
    /// `[rows, hidden]` in `moe_output`, gate/up in the layer's usual
    /// `expert_gate_out` / `expert_up_out` scratch.
    pub(super) fn forward_w4_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // The FFN family's cuBLASLt flag is read in `dense_ffn.rs` (one home per family).
        let cublas = super::ffn_takes_cublas(ctx);
        self.forward_w4_rows_with(input, rows, ctx, cublas, stream)
    }

    /// [`Self::forward_w4_rows`] with the fallback GEMM's cuBLASLt choice
    /// made by the caller (tests pin the in-tree ladder with `false`).
    pub(super) fn forward_w4_rows_with(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        cublas: bool,
        stream: u64,
    ) -> Result<()> {
        let Some(w) = self.w4.as_ref() else {
            anyhow::bail!("dense FFN W4 dispatch reached without W4 weights installed");
        };
        let (h, inter) = (ctx.config.hidden_size, ctx.config.intermediate_size);
        select::check_ffn(&*w.gate, &*w.up, &*w.down, h, inter)?;
        if rows == 0 {
            return Ok(());
        }
        let capturing = ctx.graph_capture || ctx.gpu.stream_is_capturing(stream);
        let s = w.scratch(ctx.gpu, capturing, stream)?;
        if ctx.stats.once("log:ffn_w4_dispatch") {
            tracing::info!(
                "[avarok] dense FFN W4: {} gate/up/down ({} MiB/layer), decode rows <= {}, \
                 prefill through W4Linear::prefill with the BF16 dequant fallback",
                w.gate.format(),
                w.device_bytes() >> 20,
                w.gate.max_decode_rows().min(w.up.max_decode_rows()),
            );
        }
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        // Port row `int8_prefill` (L17b): the fused int8 prompt FFN (SwiGLU, group
        // sums and quantization in one pass), byte for byte what the per-projection
        // path below gives. `false` = declined with nothing launched (row off, rows
        // within the decode window, weights not live 4-bit group-64 affine, kernels
        // or scratch absent); the arm below then runs unchanged. SiLU layers only:
        // the fused kernel is `moe_silu_mul`'s expression.
        if self.activation == super::FfnActivation::SiLU {
            let io = crate::port::affine::prefill::FfnIo {
                x: input,
                rows,
                gate_out,
                up_out,
                out: ctx.buffers.moe_output(),
            };
            if crate::port::affine::prefill::ffn_int8(
                &*w.gate, &*w.up, &*w.down, ctx.gpu, io, &s, stream,
            )? {
                return Ok(());
            }
        }
        let p = Proj {
            ctx,
            s: &s,
            spec: &w.spec,
            cublas,
            stream,
        };
        match select::pair_call(rows, w.gate.max_decode_rows(), w.up.max_decode_rows()) {
            PairCall::Pair => w
                .gate
                .decode_pair(&*w.up, ctx.gpu, input, rows, gate_out, up_out, &s, stream)?,
            PairCall::Separate => {
                self.w4_proj(&p, &*w.gate, input, rows, gate_out)?;
                self.w4_proj(&p, &*w.up, input, rows, up_out)?;
            }
        }
        let elems = u32::try_from(rows * inter)
            .map_err(|_| anyhow::anyhow!("dense FFN W4: {rows} x {inter} elements overflow u32"))?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            elems,
            stream,
        )?;
        self.w4_proj(&p, &*w.down, gate_out, rows, ctx.buffers.moe_output())
    }

    /// One projection: decode, or chunked prefill with the BF16 fallback.
    fn w4_proj(
        &self,
        p: &Proj<'_, '_>,
        w: &dyn W4Linear,
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
    ) -> Result<()> {
        let gpu = p.ctx.gpu;
        if select::proj_call(rows, w.max_decode_rows()) == ProjCall::Decode {
            return w.decode(gpu, x, rows, out, p.s, p.stream);
        }
        let (n, k) = (w.n(), w.k());
        let cap = select::prefill_chunk_cap(w.max_decode_rows());
        // The projection's BF16 copy, made on the first declined chunk only.
        let mut dequant: Option<DevicePtr> = None;
        for (first, len) in select::prefill_chunks(rows, cap, w.max_decode_rows()) {
            let xc = x.offset(first * k * 2);
            let oc = out.offset(first * n * 2);
            if w.prefill(gpu, xc, len, oc, p.s, p.stream)? {
                continue;
            }
            w4::note_prefill_decline();
            if p.ctx.stats.once("log:ffn_w4_prefill_declined") {
                tracing::info!(
                    "[avarok] dense FFN W4: {} prefill declined (first seen at {len} rows); \
                     that chunk runs the BF16 dequant + GEMM fallback",
                    w.format()
                );
            }
            let wb = match dequant {
                Some(ptr) => ptr,
                None => {
                    let ptr = scratch::bf16_dequant(gpu, p.spec)?;
                    w.dequant_bf16(gpu, ptr, p.stream)?;
                    dequant = Some(ptr);
                    ptr
                }
            };
            self.w4_bf16_gemm(p, xc, wb, oc, len, n, k)?;
        }
        Ok(())
    }

    /// The layer's BF16 prefill GEMM ladder over a dequantized `[n, k]`
    /// weight: the same three rungs, in the same order, as the native BF16
    /// overlay's `ffn_gemm!` in `forward_prefill_inner`.
    #[allow(clippy::too_many_arguments)]
    fn w4_bf16_gemm(
        &self,
        p: &Proj<'_, '_>,
        a: DevicePtr,
        w: DevicePtr,
        c: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let (m, n, k) = (m as u32, n as u32, k as u32);
        if p.cublas {
            return ops::cublas_bf16_proj_dense(a, w, c, m, n, k, p.stream);
        }
        anyhow::ensure!(
            self.dense_gemm_tc_k.0 != 0 || self.dense_gemm_bf16_k.0 != 0,
            "dense FFN W4: a declined prefill needs a BF16 GEMM, and this target resolved \
             neither cuBLASLt for the FFN nor `dense_gemm_tc` / `dense_gemm_bf16`"
        );
        let dw = DenseWeight { weight: w };
        if self.dense_gemm_tc_k.0 != 0 {
            ops::dense_gemm_tc(
                p.ctx.gpu,
                self.dense_gemm_tc_k,
                a,
                &dw,
                c,
                m,
                n,
                k,
                p.stream,
            )
        } else {
            ops::dense_gemm(
                p.ctx.gpu,
                self.dense_gemm_bf16_k,
                a,
                &dw,
                c,
                m,
                n,
                k,
                p.stream,
            )
        }
    }
}

/// What every projection call of one forward shares.
struct Proj<'a, 'c> {
    ctx: &'a ForwardContext<'c>,
    s: &'a W4Scratch,
    spec: &'a ScratchSpec,
    cublas: bool,
    stream: u64,
}

#[cfg(test)]
#[path = "w4_tests.rs"]
mod tests;
