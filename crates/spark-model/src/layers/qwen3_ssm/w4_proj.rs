// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the packed-weight (`W4Linear` contract)
//! arm of the GDN layer's two large projections, `in_proj_qkvz`
//! (`[qkvz, hidden]`) and `out_proj` (`[hidden, value]`): installing the pair
//! on a layer, reserving its scratch, and the one dispatch every projection
//! site of the layer calls.
//!
//! # What
//!
//! A loader route (lane L23a's `gdn_affine4_from_bf16`, or a packed install
//! by lane L23b) quantizes the two projections and calls
//! [`Qwen3SsmLayer::set_w4_proj`]. From then on the pair is the ONLY live copy
//! of both weights: the loader builds such a layer like the native-FP8 GDN arm
//! (`in_proj_qkvz` NULL, `ssm.out_proj` null, no NVFP4 copies, no
//! `out_proj_dense`), the setter drops any BF16 slot it still finds, and every
//! projection site takes the W4 arm FIRST:
//!
//! | site | rows | W4 call |
//! |---|---|---|
//! | M = 1 decode (`ssm_forward.rs`) | 1 | `decode` |
//! | batched decode / MTP verify (`trait_decode_batched.rs`) | sum of ks | `decode` in chunks of `max_decode_rows` |
//! | multi-sequence decode (`trait_decode_multi_seq/ssm_batched_proj.rs`) | n | `decode` in chunks of `max_decode_rows` |
//! | prefill (`trait_prefill_proj.rs`, `trait_prefill_helper.rs`) | m | `decode` at m <= `max_decode_rows`, else `prefill` chunks |
//!
//! The row plan and why decode-side rows never reach `prefill` are in
//! `w4_proj_plan.rs`. A declined `prefill` chunk (`Ok(false)`) dequantizes
//! that weight ONCE per call into the shared BF16 fallback region and runs a
//! BF16 GEMM on the chunk (the layer's tensor-core
//! `dense_gemm_bf16_pipelined`, else cuBLASLt, else scalar `dense_gemm`).
//!
//! # Off by default
//!
//! Nothing here runs unless a loader installed a pair, which only happens
//! behind a route row that is `false` on every target (`gdn_affine4_from_bf16`
//! and the packed rows). Without a pair every site's first branch is false and
//! the existing dispatch runs exactly as before, on every target.
//!
//! # Scratch
//!
//! The `W4Scratch` regions are the shared, backend-scoped ones of
//! `port::w4::scratch` (also used by the dense FFN and the attention arm).
//! [`Qwen3SsmLayer::prepare_w4_proj`] reserves them at load, before the
//! KV cache is sized and before any graph capture; the first eager forward
//! reserves them too. A capture that reaches an unreserved layer is refused
//! with the fix in the message instead of allocating inside the capture.

use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::port::w4::scratch::{self, ScratchSpec};
use crate::port::w4::{self, W4Box, W4Format, W4Linear, W4Scratch};
use crate::weight_map::DenseWeight;

#[path = "w4_proj_plan.rs"]
mod plan;
use plan::{W4At, W4Role, W4Step, check_pair, check_pair_against};

#[path = "w4_proj_gemm.rs"]
mod gemm;

#[cfg(test)]
#[path = "w4_proj_fake.rs"]
mod fake;
#[cfg(test)]
#[path = "w4_proj_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "w4_proj_tests_errors.rs"]
mod tests_errors;

/// Bytes per BF16 element.
const BF16: usize = 2;

/// The installed pair and the scratch view reserved for it.
pub(super) struct GdnW4Proj {
    qkvz: W4Box,
    out: W4Box,
    spec: ScratchSpec,
    /// Set by `prepare_w4_proj` or the first eager call; read by every
    /// call after, so a captured decode needs no host-side cache lookup.
    scratch: OnceLock<W4Scratch>,
}

impl GdnW4Proj {
    fn new(qkvz: W4Box, out: W4Box) -> Self {
        let spec = ScratchSpec::for_all(&[&*qkvz, &*out]);
        Self {
            qkvz,
            out,
            spec,
            scratch: OnceLock::new(),
        }
    }

    fn weight(&self, role: W4Role) -> &dyn W4Linear {
        match role {
            W4Role::Qkvz => &*self.qkvz,
            W4Role::OutProj => &*self.out,
        }
    }

    /// The reserved scratch view. Reserving may allocate, which a CUDA graph
    /// capture must not do, so a capture reaching an unreserved pair is
    /// refused with the fix named.
    fn scratch(&self, gpu: &dyn GpuBackend, capturing: bool, stream: u64) -> Result<W4Scratch> {
        if let Some(s) = self.scratch.get() {
            return Ok(*s);
        }
        ensure!(
            !capturing,
            "GDN W4 projections ({}): scratch not reserved before CUDA graph capture; the \
             loader must call Qwen3SsmLayer::prepare_w4_proj after set_w4_proj",
            self.qkvz.format()
        );
        let s = scratch::acquire(gpu, &self.spec, stream)?;
        let _ = self.scratch.set(s);
        Ok(*self.scratch.get().expect("set above"))
    }
}

impl Qwen3SsmLayer {
    /// Install the packed `in_proj_qkvz` (`[ssm_qkvz_size, hidden]`, rows in
    /// the layer's QKVZ order: sequential `[Q|K|V|Z]` on a sequential layer,
    /// the checkpoint's interleaved order otherwise) and `out_proj`
    /// (`[hidden, value_dim]`). From here every projection
    /// site of the layer serves them (module docs).
    ///
    /// Returns the bytes of the BF16 slots the layer held and no longer
    /// references (the dense `in_proj_qkvz` and `out_proj_dense`), which the
    /// caller owns and may free; 0 for a layer built with NULL BF16 slots, as
    /// the loader routes build it. An `Err` names the refusal and leaves the
    /// layer exactly as it was: a pair already installed, a quantized sidecar
    /// of either projection (FP8, row-wise FP8, NVFP4, packed Q2: the loader's
    /// FP8-wins rule means those layers never get W4), or a pair whose shapes
    /// cannot be a GDN layer's. The model-config shapes are checked at the
    /// first forward, where the config is. Call [`Self::prepare_w4_proj`]
    /// next, at load.
    pub fn try_set_w4_proj(&mut self, in_proj_qkvz: W4Box, out_proj: W4Box) -> Result<usize> {
        let (q, o) = (&*in_proj_qkvz, &*out_proj);
        if let Some(why) = self.w4_install_refusal(q, o) {
            anyhow::bail!("GDN layer: packed in_proj_qkvz/out_proj not installed ({why})");
        }
        let qkvz_bytes = if self.ssm.in_proj_qkvz.weight.is_null() {
            0
        } else {
            q.n() * q.k() * BF16
        };
        let out_bytes = match self.out_proj_dense {
            Some(d) if !d.weight.is_null() => o.n() * o.k() * BF16,
            _ => 0,
        };
        self.ssm.in_proj_qkvz = DenseWeight {
            weight: DevicePtr::NULL,
        };
        self.out_proj_dense = None;
        self.w4_proj = Some(GdnW4Proj::new(in_proj_qkvz, out_proj));
        Ok(qkvz_bytes + out_bytes)
    }

    /// [`Self::try_set_w4_proj`] in the 3.7.2 form: the freed BF16 bytes, or
    /// 0 when the install is refused (logged; the layer keeps its weights). A
    /// loader that freed the BF16 sources before the install has nothing to
    /// fall back to and should call `try_set_w4_proj` and fail the load on
    /// `Err`.
    pub fn set_w4_proj(&mut self, in_proj_qkvz: W4Box, out_proj: W4Box) -> usize {
        match self.try_set_w4_proj(in_proj_qkvz, out_proj) {
            Ok(freed) => freed,
            Err(e) => {
                if self.w4_proj.is_none()
                    && self.ssm.in_proj_qkvz.weight.is_null()
                    && self.w4_sidecar().is_none()
                {
                    tracing::error!("[avarok] {e:#}; the layer has no QKVZ weight left");
                } else {
                    tracing::warn!("[avarok] {e:#}; the layer keeps its weights");
                }
                0
            }
        }
    }

    /// Why a pair cannot be installed on this layer, or `None`.
    fn w4_install_refusal(&self, q: &dyn W4Linear, o: &dyn W4Linear) -> Option<String> {
        if self.w4_proj.is_some() {
            return Some("already installed".to_string());
        }
        if let Some(which) = self.w4_sidecar() {
            return Some(format!(
                "the layer already carries a quantized {which} weight"
            ));
        }
        check_pair(q.n(), q.k(), o.n(), o.k())
            .err()
            .map(|e| e.to_string())
    }

    /// The first quantized copy of either projection the layer holds, if any.
    fn w4_sidecar(&self) -> Option<&'static str> {
        let present = [
            (self.qkvz_q2.is_some(), "packed Q2 QKVZ"),
            (self.qkvz_fp8w.is_some(), "FP8 QKVZ"),
            (self.qkvz_fp8w_rowwise.is_some(), "row-wise FP8 QKVZ"),
            (self.qkvz_fp8.is_some(), "FP8 QKVZ"),
            (self.qkvz_nvfp4.is_some(), "NVFP4 QKVZ"),
            (self.qkvz_nvfp4_t.is_some(), "NVFP4 QKVZ"),
            (self.out_proj_fp8w.is_some(), "FP8 out_proj"),
            (
                self.out_proj_fp8w_rowwise.is_some(),
                "row-wise FP8 out_proj",
            ),
            (self.out_proj_fp8.is_some(), "FP8 out_proj"),
            (self.out_proj_nvfp4_t.is_some(), "NVFP4 out_proj"),
            (!self.ssm.out_proj.weight.is_null(), "NVFP4 out_proj"),
        ];
        present.into_iter().find(|p| p.0).map(|p| p.1)
    }

    /// Whether a packed pair serves this layer's two projections.
    pub fn w4_proj_installed(&self) -> bool {
        self.w4_proj.is_some()
    }

    /// Resident device bytes of the installed pair (0 without one).
    pub fn w4_proj_device_bytes(&self) -> usize {
        self.w4_proj
            .as_ref()
            .map_or(0, |p| p.qkvz.device_bytes() + p.out.device_bytes())
    }

    /// What the pair's calls need from the shared scratch (residency and
    /// preflight sizing); `None` without a pair.
    pub fn w4_proj_scratch_spec(&self) -> Option<ScratchSpec> {
        self.w4_proj.as_ref().map(|p| p.spec)
    }

    /// The formats of `(in_proj_qkvz, out_proj)`; `None` without a pair.
    pub fn w4_proj_formats(&self) -> Option<(W4Format, W4Format)> {
        self.w4_proj
            .as_ref()
            .map(|p| (p.qkvz.format(), p.out.format()))
    }

    /// Reserve the shared W4 scratch for this layer at load, before the KV
    /// cache is sized and before any graph capture. Idempotent. Returns the
    /// bytes reserved for this layer's spec (an upper bound on what it adds:
    /// the regions are shared by every W4 consumer on the backend); 0 without
    /// a pair. Without this call the first eager forward reserves them. Same
    /// name and contract as the attention arm's (lane L22b).
    pub fn prepare_w4_proj(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<usize> {
        let Some(p) = self.w4_proj.as_ref() else {
            return Ok(0);
        };
        p.scratch(gpu, false, stream)?;
        Ok(p.spec.device_bytes())
    }

    /// The name the loader's GDN arm calls (`w4_install.rs`, lane L23b): reserve
    /// the shared W4 scratch before any capture.
    /// Exactly [`Self::prepare_w4_proj`]; the bytes are of no use to the loader.
    pub fn finalize_w4_proj_load(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        self.prepare_w4_proj(gpu, stream).map(|_| ())
    }

    /// M = 1 decode QKVZ into the sequential `[Q|K|V|Z]` buffer.
    pub(super) fn w4_qkvz_decode(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        deinterleaved: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.w4_qkvz_deinterleaved(ctx, normed, deinterleaved, 1, W4At::Rows, stream)
    }

    /// Prefill QKVZ of `m` tokens into the sequential `[Q|K|V|Z]` buffer.
    pub(super) fn w4_qkvz_prefill(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        deinterleaved: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        self.w4_qkvz_deinterleaved(ctx, normed, deinterleaved, m, W4At::Prefill, stream)
    }

    /// Decode-side QKVZ of `rows` rows into `proj_dst`; the caller owns the
    /// deinterleave, as its other arms do.
    pub(super) fn w4_qkvz_rows(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        proj_dst: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        self.w4_project(
            ctx,
            W4Role::Qkvz,
            normed,
            proj_dst,
            rows,
            W4At::Rows,
            stream,
        )
    }

    /// Decode-side `out_proj` of `rows` rows (M = 1 included).
    pub(super) fn w4_out_rows(
        &self,
        ctx: &ForwardContext,
        x: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        self.w4_project(ctx, W4Role::OutProj, x, out, rows, W4At::Rows, stream)
    }

    /// Prefill `out_proj` of `m` tokens.
    pub(super) fn w4_out_prefill(
        &self,
        ctx: &ForwardContext,
        x: DevicePtr,
        out: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        self.w4_project(ctx, W4Role::OutProj, x, out, m, W4At::Prefill, stream)
    }

    /// QKVZ into `deinterleaved`: straight in on a sequential layout, else
    /// through the `ssm_qkvz` scratch and one deinterleave launch over all
    /// rows, exactly where the BF16 and NVFP4 arms place it.
    fn w4_qkvz_deinterleaved(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        deinterleaved: DevicePtr,
        rows: usize,
        at: W4At,
        stream: u64,
    ) -> Result<()> {
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        self.w4_project(ctx, W4Role::Qkvz, normed, proj_dst, rows, at, stream)?;
        if !self.sequential_qkvz && rows > 0 {
            let c = ctx.config;
            let nk = c.linear_num_key_heads;
            ops::deinterleave_qkvz(
                ctx.gpu,
                self.deinterleave_k,
                proj_dst,
                deinterleaved,
                rows as u32,
                nk as u32,
                c.linear_key_head_dim as u32,
                (c.linear_num_value_heads / nk.max(1)) as u32,
                c.linear_value_head_dim as u32,
                stream,
            )?;
        }
        Ok(())
    }

    /// The W4 projection: `out[rows, n] = x[rows, k] @ W^T`, both BF16 and
    /// row-major, by the row plan of `w4_proj_plan.rs`.
    #[allow(clippy::too_many_arguments)]
    fn w4_project(
        &self,
        ctx: &ForwardContext,
        role: W4Role,
        x: DevicePtr,
        out: DevicePtr,
        rows: usize,
        at: W4At,
        stream: u64,
    ) -> Result<()> {
        let p = self
            .w4_proj
            .as_ref()
            .ok_or_else(|| anyhow!("{}: W4 arm reached with no pair installed", role.what()))?;
        check_pair_against(&*p.qkvz, &*p.out, ctx.config)?;
        ensure!(
            ctx.config.tp_world_size <= 1,
            "{}: the W4 pair is a whole (unsharded) weight and this model runs \
             tensor-parallel over {} ranks; the loader routes install W4 on one GPU only",
            role.what(),
            ctx.config.tp_world_size
        );
        if rows == 0 {
            return Ok(());
        }
        let capturing = ctx.graph_capture || ctx.gpu.stream_is_capturing(stream);
        let s = p.scratch(ctx.gpu, capturing, stream)?;
        self.w4_log_route(ctx, p);
        let w = p.weight(role);
        let (n, k) = (w.n(), w.k());
        // The projection's BF16 copy, made on its first declined chunk only.
        let mut dequant: Option<DevicePtr> = None;
        for c in plan::plan(at, rows, w.max_decode_rows()) {
            let xc = x.offset(c.start * k * BF16);
            let oc = out.offset(c.start * n * BF16);
            if c.step == W4Step::Decode {
                w.decode(ctx.gpu, xc, c.len, oc, &s, stream)
                    .with_context(|| format!("{} W4 decode of {} rows", role.what(), c.len))?;
                continue;
            }
            let done = w
                .prefill(ctx.gpu, xc, c.len, oc, &s, stream)
                .with_context(|| format!("{} W4 prefill of {} rows", role.what(), c.len))?;
            if done {
                continue;
            }
            w4::note_prefill_decline();
            if ctx.stats.once(role.decline_key()) {
                tracing::info!(
                    "[avarok] {} W4 prefill declined (first seen at {} rows, {} [{n}x{k}]): \
                     that chunk runs the BF16 dequant + GEMM fallback; said once per model",
                    role.what(),
                    c.len,
                    w.format()
                );
            }
            let wb = match dequant {
                Some(ptr) => ptr,
                None => {
                    let ptr = scratch::bf16_dequant(ctx.gpu, &p.spec)?;
                    w.dequant_bf16(ctx.gpu, ptr, stream)
                        .with_context(|| format!("{} W4 dequant to BF16", role.what()))?;
                    dequant = Some(ptr);
                    ptr
                }
            };
            self.w4_bf16_gemm(ctx, xc, wb, oc, c.len, n, k, stream)?;
        }
        Ok(())
    }
}
