// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the packed-weight (W4) q/k/v/o projections of a full-attention layer (port lane
//! L22b, the `W4Linear` contract): the per-layer slot set [`AttnW4`], the installers
//! [`Qwen3AttentionLayer::try_set_w4_proj`] and [`Qwen3AttentionLayer::set_w4_proj`], the load-time
//! scratch reservation [`Qwen3AttentionLayer::prepare_w4_proj`], and the install rules. The
//! dispatch helpers every q/k/v/o call site runs live in the child `w4_proj_dispatch.rs`.
//!
//! # Where the arms sit
//!
//! Every projection chain of the layer checks its W4 slot FIRST and otherwise runs exactly the code
//! it ran before, so a layer without W4 weights is byte-identical to the base commit:
//!
//! | path | q | k, v | o |
//! |---|---|---|---|
//! | M=1 decode | `decode/attention_forward.rs` (lane L29's file, requested hunk) | `decode/attention_forward_kv.rs` | `decode/attention_forward_oproj.rs` |
//! | batched decode and verify (any `n`) | `trait_impl/multi_seq/qkv.rs` -> `w4_qkv_strided` | same | `trait_impl/multi_seq/attn/o_proj.rs` |
//! | prefill chunk 0 | `prefill/cache_skip_qkv.rs` | same | `prefill/paged_oproj.rs` |
//! | prefill later chunks | `prefill/paged_qkv.rs` | same | `prefill/paged_oproj.rs` |
//!
//! Decode and verify always take [`W4Linear::decode`] in chunks of at most `max_decode_rows()` rows.
//! The contract makes every output row's bits independent of the row count, so an MTP verify row
//! equals the M=1 decode of the same token and speculative decoding at temperature 0 stays exact.
//! Prefill takes [`W4Linear::prefill`] in the chunks `port::w4::select::prefill_chunks` cuts; a
//! declined chunk is recomputed from the BF16 dequant of the weight through the layer's BF16 GEMM.
//!
//! # What an install changes
//!
//! An install takes all four projections or none, only on a layer with no MLA, no K=V sharing and
//! no quantized sidecar of any kind. It accepts both layer shapes a loader builds: BF16 q/k/v and
//! the BF16 O-projection still present (they are NULLed, and their bytes are returned for the
//! caller to free), or those slots already empty because the loader quantized and freed each
//! source first (the BF16 -> FP8 route's shape; 0 bytes returned). A reader that was missed then
//! fails on a NULL pointer instead of reading freed memory. LoRA folds, the head gate, the q/k
//! norms and the gated `[Q|gate]` split are untouched: the W4 arm writes the same raw interleaved
//! layout the BF16 GEMV and GEMM write.

use std::sync::OnceLock;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Qwen3AttentionLayer;
use crate::port::w4::scratch::{self, ScratchSpec};
use crate::port::w4::{W4Box, W4Linear, W4Scratch};
use crate::weight_map::DenseWeight;

/// Rows of the batched-decode staging region: the multi-sequence q/k/v land here contiguous and
/// are scattered into the per-sequence `[Q | K | V]` layout with one pitched copy each. A batch
/// wider than this walks in blocks of this many rows.
pub(crate) const STAGE_ROWS: usize = 64;
const TAG_STAGE: &str = "port.w4.attn.stage";
const BF16: usize = 2;

/// One of the four attention projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum W4Slot {
    Q,
    K,
    V,
    O,
}

/// The device regions one installed layer uses, fixed once reserved (graph-capture safe).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct W4Ready {
    /// The shared `port::w4` scratch view for this layer's four weights.
    pub s: W4Scratch,
    /// What `s` was reserved for (also sizes the BF16 decline fallback).
    pub spec: ScratchSpec,
    /// `[STAGE_ROWS, q_n + k_n + v_n]` BF16 staging for the batched q/k/v scatter.
    pub stage: DevicePtr,
    pub stage_bytes: usize,
}

/// The W4 projections of one attention layer. Empty by default: every arm then falls through.
#[derive(Default)]
pub(crate) struct AttnW4 {
    q: Option<W4Box>,
    k: Option<W4Box>,
    v: Option<W4Box>,
    o: Option<W4Box>,
    ready: OnceLock<W4Ready>,
}

impl AttnW4 {
    pub(crate) fn get(&self, slot: W4Slot) -> Option<&W4Box> {
        match slot {
            W4Slot::Q => self.q.as_ref(),
            W4Slot::K => self.k.as_ref(),
            W4Slot::V => self.v.as_ref(),
            W4Slot::O => self.o.as_ref(),
        }
    }

    /// All four are installed together, so one slot answers for the set.
    pub(crate) fn is_installed(&self) -> bool {
        self.q.is_some()
    }

    fn all(&self) -> Vec<&dyn W4Linear> {
        [&self.q, &self.k, &self.v, &self.o]
            .into_iter()
            .flatten()
            .map(|w| w.as_ref())
            .collect()
    }

    /// Resident bytes of the packed weights.
    pub(crate) fn device_bytes(&self) -> usize {
        self.all().iter().map(|w| w.device_bytes()).sum()
    }

    /// The `port::w4` scratch the four weights need.
    pub(crate) fn spec(&self) -> ScratchSpec {
        ScratchSpec::for_all(&self.all())
    }

    /// Bytes of the batched q/k/v staging region.
    pub(crate) fn stage_bytes(&self) -> usize {
        let width: usize = [&self.q, &self.k, &self.v]
            .into_iter()
            .flatten()
            .map(|w| w.n())
            .sum();
        STAGE_ROWS * width * BF16
    }
}

/// Contiguous `(first_row, rows)` chunks of at most `max` rows covering `rows` (decode calls).
pub(crate) fn decode_chunks(rows: usize, max: usize) -> impl Iterator<Item = (usize, usize)> {
    let max = max.max(1);
    (0..rows)
        .step_by(max)
        .map(move |first| (first, (rows - first).min(max)))
}

/// Why an install must not put these weights on a layer with this shape, if anything.
/// `gated`: q emits interleaved `[Q|gate]`, so q has twice o's input width.
pub(crate) fn shape_refusal(
    gated: bool,
    q: &dyn W4Linear,
    k: &dyn W4Linear,
    v: &dyn W4Linear,
    o: &dyn W4Linear,
) -> Option<String> {
    let hidden = o.n();
    let q_dim = o.k();
    let want_q = q_dim * if gated { 2 } else { 1 };
    if q.k() != hidden || k.k() != hidden || v.k() != hidden {
        return Some(format!(
            "q/k/v input widths {}/{}/{} differ from o_proj's output width {hidden}",
            q.k(),
            k.k(),
            v.k()
        ));
    }
    if k.n() != v.n() {
        return Some(format!(
            "k_proj is [{}, _] but v_proj is [{}, _]",
            k.n(),
            v.n()
        ));
    }
    if q.n() != want_q {
        return Some(format!(
            "q_proj emits {} features, the layer needs {want_q} (o_proj input {q_dim}, gated {gated})",
            q.n()
        ));
    }
    // 16-byte rows for every input (`cp.async` loads) and 4-byte rows for every output.
    let ws = [q, k, v, o];
    if let Some(w) = ws
        .iter()
        .find(|w| !w.k().is_multiple_of(8) || !w.n().is_multiple_of(2))
    {
        return Some(format!(
            "a [{}, {}] {} weight leaves its rows unaligned (k % 8, n % 2)",
            w.n(),
            w.k(),
            w.format()
        ));
    }
    if ws.iter().any(|w| w.max_decode_rows() == 0) {
        return Some("a weight reports max_decode_rows 0".to_string());
    }
    None
}

impl Qwen3AttentionLayer {
    /// Install packed q/k/v/o weights. Returns the BF16 bytes the layer no longer
    /// reads, which the caller may free (0 when the loader already freed them). An `Err` names the
    /// refusal and leaves the layer exactly as it was: already installed, an MLA or K=V layer, a
    /// quantized sidecar of any kind, or projection shapes that do not fit together. Call
    /// [`Self::prepare_w4_proj`] next, at load.
    pub fn try_set_w4_proj(&mut self, q: W4Box, k: W4Box, v: W4Box, o: W4Box) -> Result<usize> {
        if let Some(why) = self.w4_install_refusal(&*q, &*k, &*v, &*o) {
            anyhow::bail!(
                "attention layer {}: packed q/k/v/o not installed ({why})",
                self.attn_layer_idx
            );
        }
        let present = |p: DevicePtr| !p.is_null();
        let mut freed = 0;
        if present(self.attn.q_proj.weight) {
            freed += q.n() * q.k() * BF16;
        }
        if present(self.attn.k_proj.weight) {
            freed += k.n() * k.k() * BF16;
        }
        if present(self.attn.v_proj.weight) {
            freed += v.n() * v.k() * BF16;
        }
        if self.o_dense_bf16.is_some() {
            freed += o.n() * o.k() * BF16;
        }
        if self.attn_layer_idx == 0 {
            tracing::info!(
                "[avarok] attention W4: q/k/v/o {} / {} / {} / {} installed ({} BF16 bytes released \
                 per layer); decode and verify take W4 decode (row invariant), prefill W4 prefill",
                q.format(),
                k.format(),
                v.format(),
                o.format(),
                freed
            );
        }
        let null = DenseWeight {
            weight: DevicePtr::NULL,
        };
        self.attn.q_proj = null;
        self.attn.k_proj = null;
        self.attn.v_proj = null;
        self.o_dense_bf16 = None;
        self.w4_proj = AttnW4 {
            q: Some(q),
            k: Some(k),
            v: Some(v),
            o: Some(o),
            ready: OnceLock::new(),
        };
        Ok(freed)
    }

    /// [`Self::try_set_w4_proj`] in the 3.7.2 form: the freed BF16 bytes, or 0 when the install is
    /// refused (logged). A refused layer keeps its weights. A loader that freed the BF16 sources
    /// before the install has nothing to fall back to and should call `try_set_w4_proj` and fail
    /// the load on `Err`.
    pub fn set_w4_proj(&mut self, q: W4Box, k: W4Box, v: W4Box, o: W4Box) -> usize {
        match self.try_set_w4_proj(q, k, v, o) {
            Ok(freed) => freed,
            Err(e) => {
                if self.attn.q_proj.weight.is_null() && self.q_weight.is_none() {
                    tracing::error!("[avarok] {e}; the layer has no q/k/v/o weights left");
                } else {
                    tracing::warn!("[avarok] {e}; the layer keeps its weights");
                }
                0
            }
        }
    }

    fn w4_install_refusal(
        &self,
        q: &dyn W4Linear,
        k: &dyn W4Linear,
        v: &dyn W4Linear,
        o: &dyn W4Linear,
    ) -> Option<String> {
        if self.w4_proj.is_installed() {
            return Some("already installed".to_string());
        }
        if self.mla.is_some() {
            return Some("MLA layer".to_string());
        }
        if self.k_eq_v {
            return Some("K=V shared projection".to_string());
        }
        let sidecar = self.q_weight.is_some()
            || self.k_weight.is_some()
            || self.v_weight.is_some()
            || self.o_weight.is_some()
            || !self.attn.o_proj.is_null()
            || self.qkv_nvfp4_t.is_some()
            || self.q_nvfp4_t.is_some()
            || self.k_nvfp4_t.is_some()
            || self.v_nvfp4_t.is_some()
            || self.o_nvfp4_t.is_some()
            || self.q_fp8w_t.is_some()
            || self.k_fp8w_t.is_some()
            || self.v_fp8w_t.is_some()
            || self.o_fp8w_t.is_some()
            || self.q_fp8.is_some()
            || self.k_fp8.is_some()
            || self.v_fp8.is_some()
            || self.o_fp8.is_some();
        if sidecar {
            return Some("the layer already carries quantized q/k/v/o weights".to_string());
        }
        shape_refusal(self.gated, q, k, v, o)
    }

    /// Whether packed q/k/v/o are installed.
    pub fn w4_proj_installed(&self) -> bool {
        self.w4_proj.is_installed()
    }

    /// Resident bytes of the installed packed weights (0 when none), for the residency audit.
    pub fn w4_proj_device_bytes(&self) -> usize {
        self.w4_proj.device_bytes()
    }

    /// Reserve this layer's scratch (the shared `port::w4` regions and the staging region) now,
    /// outside graph capture and before the KV cache is sized. Returns an upper bound of the bytes
    /// it may have added (the regions are shared, so usually less); 0 when nothing is installed.
    /// Idempotent. Without this call the first eager forward reserves them.
    pub fn prepare_w4_proj(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<usize> {
        if !self.w4_proj.is_installed() {
            return Ok(0);
        }
        let r = self.w4_ready(gpu, stream)?;
        Ok(r.spec.device_bytes() + r.stage_bytes)
    }

    /// The reserved regions, reserving them on first use. Refuses during capture: allocation is
    /// not capturable, and a graph must never bake a region that a later reservation replaces.
    pub(super) fn w4_ready(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<W4Ready> {
        if let Some(r) = self.w4_proj.ready.get() {
            return Ok(*r);
        }
        ensure!(
            self.w4_proj.is_installed(),
            "attention layer {}: no packed q/k/v/o installed",
            self.attn_layer_idx
        );
        ensure!(
            !gpu.stream_is_capturing(stream),
            "attention layer {}: packed q/k/v/o scratch not reserved before CUDA graph capture; \
             the loader must call Qwen3AttentionLayer::prepare_w4_proj after the install",
            self.attn_layer_idx
        );
        let spec = self.w4_proj.spec();
        let s = scratch::acquire(gpu, &spec, stream)?;
        let stage_bytes = self.w4_proj.stage_bytes();
        let stage = gpu.op_cache().scratch(gpu, TAG_STAGE, stage_bytes)?;
        let r = W4Ready {
            s,
            spec,
            stage,
            stage_bytes,
        };
        // A racing first call reserved the same cached regions; either value serves.
        let _ = self.w4_proj.ready.set(r);
        Ok(*self.w4_proj.ready.get().unwrap_or(&r))
    }
}

/// The dispatch helpers the q/k/v/o call sites run (decode windows, the prefill decline fallback,
/// the batched q/k/v scatter).
#[path = "w4_proj_dispatch.rs"]
mod dispatch;

/// Mock-backend fixtures shared by the W4 dispatch tests of this layer (here, `prefill/` and
/// `trait_impl/multi_seq/`).
#[cfg(test)]
#[path = "w4_proj_fixture.rs"]
pub(super) mod fixture;

#[cfg(test)]
#[path = "w4_proj_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "w4_proj_dispatch_tests.rs"]
mod dispatch_tests;
